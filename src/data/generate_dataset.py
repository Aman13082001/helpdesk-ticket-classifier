"""Generate a realistic SYNTHETIC university-helpdesk ticket dataset.

Why synthetic?  Public helpdesk datasets are corporate-IT focused and do not contain
university departments (exams, hostel, fees ...).  We therefore generate tickets from a
large bank of hand-written templates, with:

* slot filling for domain entities (student IDs, course codes, rooms, error codes, amounts,
  transaction IDs, dates, people, systems) - these also exercise the NER layer;
* inline alternatives  ``[word1|word2|word3]``  so the same template yields many phrasings;
* SHARED vocabulary across departments ("portal", "password", "payment", "library", "room",
  "exam") so a single keyword cannot decide the class (no keyword leakage);
* multi-intent tickets (e.g. "fee paid but portal shows unpaid") whose label is the
  *primary* intent -> genuinely ambiguous cases for error analysis;
* short / vague tickets with weak cues;
* real-world noise: typos, abbreviations (pls, acct, wifi), casing, emojis, URLs, e-mails,
  phone numbers, repeated punctuation, elongated words;
* a small fraction of "mis-filed" labels, as found in real ticket logs;
* injected duplicates / empty rows so the cleaning step (clean_split.py) has real work.

Usage:
    python -m src.data.generate_dataset            # uses config.yaml
    python -m src.data.generate_dataset --n 3000 --out data/raw/my.csv
"""
from __future__ import annotations

import argparse
import random
import re
import string
from datetime import date, timedelta

import pandas as pd

from src.utils import get_logger, load_config, resolve, set_seed

log = get_logger("generate_dataset")

# ---------------------------------------------------------------------------
# Slot generators (entity values). Kept as functions of an RNG for reproducibility.
# ---------------------------------------------------------------------------
PROGRAMS = ["MCA", "BCA", "BTECH", "MBA", "BSC", "MTECH", "BCOM"]
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Karan", "Isha", "Rahul", "Neha",
         "Arjun", "Pooja", "Siddharth", "Kavya", "Aditya", "Meera", "Nikhil", "Riya", "Varun", "Simran"]
LAST = ["Sharma", "Verma", "Gupta", "Nair", "Iyer", "Singh", "Mehta", "Reddy", "Kapoor", "Joshi",
        "Malhotra", "Chopra", "Bansal", "Das", "Kulkarni"]
TITLES = ["Dr.", "Prof.", "Mr.", "Ms.", "Mrs."]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _sid(r: random.Random) -> str:
    return f"KRM{r.choice([2022, 2023, 2024, 2025])}{r.choice(PROGRAMS)}{r.randint(1, 180):03d}"


def _course(r: random.Random) -> str:
    return r.choice([
        f"MCA-{r.choice([101, 102, 201, 203, 301, 302, 305])}", f"CS{r.randint(101, 499)}",
        f"BCA-{r.randint(101, 399)}", f"ENCS{r.randint(101, 499)}", f"MBA-{r.randint(101, 299)}",
        "ETMMTI303", "ETMMTI301", f"ETCS{r.randint(201, 499)}", f"BTCS-{r.randint(201, 499)}",
    ])


def _date(r: random.Random) -> str:
    d = r.randint(1, 28)
    m = r.randint(7, 12)
    return r.choice([
        f"{d} {MONTHS[m - 1]}", f"{d}/{m}/2026", f"{d:02d}-{m:02d}-2026", f"{MONTHS[m - 1]} {d}",
        r.choice(WEEKDAYS), "yesterday", "last week", f"{d} {MONTHS[m - 1]} 2026",
    ])


def _err(r: random.Random) -> str:
    return r.choice([
        f"ERR_{r.choice([401, 403, 404, 500, 502, 503])}", f"E-{r.randint(1000, 1999)}",
        f"HTTP {r.choice([400, 403, 404, 500, 502, 504])}", f"ERR_{r.choice(['TIMEOUT', 'AUTH', 'SESSION'])}",
        f"error code {r.randint(100, 999)}",
    ])


def _amount(r: random.Random) -> str:
    v = r.choice([500, 1200, 2500, 3200, 4500, 7800, 12500, 15000, 22000, 45000, 58500, 72000])
    return r.choice([f"Rs {v:,}", f"Rs. {v}", f"INR {v}", f"₹{v:,}", f"₹ {v}", f"{v} rupees"])


def _txn(r: random.Random) -> str:
    return r.choice([f"TXN{r.randint(10**9, 10**10 - 1)}", f"UTR{r.randint(10**11, 10**12 - 1)}",
                     f"TXN-{r.randint(100000, 999999)}"])


def _person(r: random.Random) -> str:
    return f"{r.choice(TITLES)} {r.choice(FIRST)} {r.choice(LAST)}" if r.random() < 0.6 \
        else f"{r.choice(FIRST)} {r.choice(LAST)}"


SLOTS = {
    "sid": _sid,
    "course": _course,
    "date": _date,
    "err": _err,
    "amt": _amount,
    "txn": _txn,
    "person": _person,
    "block": lambda r: f"Block {r.choice('ABCDEF')}",
    "room": lambda r: f"Hostel {r.choice('ABCD')}-{r.randint(1, 4)}{r.randint(1, 30):02d}",
    "lab": lambda r: f"Lab {r.randint(1, 9)}",
    "time": lambda r: f"{r.randint(8, 11)}:{r.choice(['00', '15', '30', '45'])} {r.choice(['am', 'AM'])}" if r.random() < .5
    else f"{r.randint(1, 6)} pm",
    "email": lambda r: f"{r.choice(FIRST).lower()}.{r.choice(LAST).lower()}{r.randint(1, 99)}@"
                       f"{r.choice(['krmu.edu.in', 'gmail.com', 'yahoo.co.in', 'outlook.com'])}",
    "phone": lambda r: r.choice([f"9{r.randint(100000000, 999999999)}", f"+91 9{r.randint(1000, 9999)} {r.randint(10000, 99999)}"]),
    "url": lambda r: r.choice([
        f"https://lms.krmu.edu.in/course/view.php?id={r.randint(100, 999)}",
        "https://portal.krmu.edu.in/student/login", f"https://fees.krmu.edu.in/pay?ref={r.randint(1000, 9999)}",
        "http://exam.krmu.edu.in/hallticket", "www.krmu.edu.in/hostel"]),
    "n": lambda r: str(r.randint(2, 6)),
    "sem": lambda r: r.choice(["1st", "2nd", "3rd", "4th", "5th", "6th", "III", "first", "third"]),
    "ticket": lambda r: f"#{r.randint(10000, 99999)}",
}

# ---------------------------------------------------------------------------
# Core problem templates per category ([a|b] = random alternative, {slot} = entity)
# ---------------------------------------------------------------------------
CORE: dict[str, list[str]] = {
    "IT_WiFi": [
        "[wifi|wi-fi|WiFi|internet] is [not working|very slow|keeps disconnecting] in {block} since {date}",
        "I am unable to connect to Eduroam on my laptop, it says authentication failed",
        "my university email account is locked after too many wrong password attempts",
        "need to reset the password of my official email id, the reset link [never arrives|has expired]",
        "VPN is not connecting from home, getting {err} every time",
        "the PCs in {lab} are not booting, students cannot do the {course} practical",
        "MS Office licence on my laptop shows [expired|product deactivated], please renew it",
        "projector and system in {block} room {n}0{n} are not working before the lecture",
        "cannot access Microsoft Teams with my university id, it says [account not found|{err}]",
        "the library computers cannot connect to the internet, page does not load",
        "[wifi|internet] in my hostel room is [dead|extremely slow|not working] at night, cannot attend online class",
        "printer on the {block} ground floor is offline and the queue is stuck",
        "MATLAB / Python software is not installed on the systems in {lab}",
        "my laptop is not getting an IP address on the campus network",
        "LAN port in {block} faculty cabin is not working",
        "two factor authentication codes for my email are not coming to {phone}",
        "need a new email id created for a newly joined faculty {person}",
        "antivirus is showing a virus alert on all computers of {lab}",
        "website {url} is showing {err} on the campus network but works on mobile data",
        "the biometric attendance device in {block} is not syncing with the server",
        "I forgot my wifi login password and the self service reset does not work",
        "Zoom and Google Meet are blocked on the campus wifi, cannot join the webinar",
        "keyboard and mouse missing from system {n} in {lab}",
        "network is down in the entire {block} since {time}",
        "unable to install the licensed SPSS software, the licence key is invalid",
        "outlook keeps asking for my password again and again and does not sync mail",
        "system in the exam hall lab is very slow, it will be a problem during the online test",
        "campus wifi portal is not opening the login page, the captive page shows {err}",
    ],
    "LMS_Portal": [
        "cannot login to the student portal, it shows [invalid credentials|{err}] even with the correct password",
        "assignment upload on LMS for {course} is failing with {err}",
        "I am not able to see the course {course} on Moodle, it is missing from my dashboard",
        "the quiz on LMS got auto submitted after {n} minutes even though time was left",
        "student portal shows wrong course enrolment, {course} is not listed",
        "LMS says submission deadline passed but it was open till today midnight",
        "my profile photo and name are wrong on the student portal",
        "password reset link for the student portal is not received on my registered email {email}",
        "video lectures on the LMS are not playing, buffering forever",
        "uploaded the wrong file for the {course} assignment on Moodle, please allow resubmission",
        "portal session expires every 2 minutes and logs me out",
        "course enrolment key for {course} on LMS is not working",
        "LMS is not showing grades for the assignments submitted in {course}",
        "the student portal page {url} is not loading after login",
        "discussion forum on the LMS is not allowing me to post",
        "cannot download the study material PDF for {course} from Moodle",
        "my student ID {sid} is not linked with the portal account",
        "elective subject selection option is not visible on the portal",
        "portal app on mobile crashes when I open the attendance section",
        "teacher {person} says the assignment was not received on LMS but I submitted it",
        "turnitin similarity report is not generating on the LMS submission",
        "my LMS account shows the previous semester courses instead of {sem} semester",
        "online feedback form on the portal is not submitting, stuck on loading",
        "notifications from the LMS are not coming to my email",
        "I cannot see the upload button for the lab record in {course}",
        "the portal shows my mobile number as {phone} which is wrong, unable to update it",
    ],
    "Academics_Exams": [
        "my hall ticket for the {sem} semester exams is not generated yet",
        "there is a clash in the timetable, {course} and another exam are on the same day {date}",
        "I want to apply for re-evaluation of my {course} answer sheet",
        "result of {course} shows absent but I appeared for the exam on {date}",
        "attendance in {course} is shown below 75 percent but I attended all classes",
        "please share the syllabus for the {sem} semester {course} paper",
        "I need to register for the backlog exam of {course}",
        "marks in the internal assessment of {course} are not updated",
        "the exam date sheet for the end term has not been released",
        "I was marked absent in the mid term of {course} due to medical reasons, can I get a re-test",
        "need a bonafide certificate and provisional degree for job joining",
        "my grade card shows wrong CGPA, please verify",
        "requesting change of elective from {course} to another subject after the add/drop date",
        "exam centre allotted is wrong on my admit card, I am a regular student",
        "lecture of {person} for {course} is not being taken for {n} weeks",
        "detained in {course} due to attendance shortage, want to know the condonation process",
        "migration certificate and transcript needed for higher studies",
        "practical exam of {course} clashes with my placement interview on {date}",
        "answer sheet photocopy requested on {date} but not received",
        "question paper in {course} had questions out of syllabus",
        "supplementary exam form date has passed, can it be extended",
        "my name is spelled wrong on the marksheet",
        "credits for the MOOC course are not transferred to my academic record",
        "class schedule for {block} room shows a lecture that has been cancelled",
        "minor project evaluation with {person} is scheduled on {date}, need rescheduling",
        "re-evaluation result was supposed to come out by {date} but nothing yet",
        "hall ticket is showing debarred because of attendance, I had a medical certificate",
    ],
    "Fees_Accounts": [
        "fee payment failed but {amt} was deducted from my account, transaction id {txn}",
        "I paid the semester fee of {amt} but the receipt has not been generated",
        "my scholarship amount has not been credited for the {sem} semester",
        "requesting refund of the hostel security deposit {amt}, I have left the hostel",
        "exam fee paid on {date} is not reflecting, transaction {txn}",
        "late fine of {amt} has been charged although I paid before the last date",
        "need a fee structure letter for an education loan from the bank",
        "the bank challan for the {sem} semester fee is not downloadable",
        "payment gateway shows {err} at the last step of fee payment",
        "double payment happened for the tuition fee, please refund {amt}",
        "fee concession for sibling is not applied in my fee statement",
        "I need the fee receipt for income tax purposes for both semesters",
        "library fine payment of {amt} was done online but still shows pending",
        "scholarship form verification is pending at the accounts office since {date}",
        "my education loan disbursement letter needs the accounts office stamp",
        "fee installment option is not available this semester, can I pay in two parts",
        "the fee paid via UPI with reference {txn} is not updated in the ledger",
        "convocation fee of {amt} was charged twice",
        "transport fee refund for the months I did not use the bus",
        "post matric scholarship status shows rejected, documents were complete",
        "caution money refund after course completion has not been received",
        "bank account details for the fee refund need to be updated",
        "fee due shows {amt} extra which I have already paid via NEFT",
        "hostel fee and mess advance were paid together in one transaction {txn}, please adjust",
        "need no-dues certificate from accounts for my degree",
    ],
    "Hostel_Facilities": [
        "there has been no water supply in {room} since {date}",
        "the fan and light in {room} are not working",
        "I want to change my room from {room} due to roommate issues",
        "mess food quality is very poor and we found insects in the dal today",
        "room allotment for the new semester has not been done yet",
        "AC in {block} classroom {n}0{n} is not working, it is very hot",
        "washroom on the {n}nd floor of the hostel is not cleaned for days",
        "the hot water geyser in {room} is not working",
        "library is closing at 8 pm, please extend the timing during exams",
        "college bus on route {n} did not come today, many students missed the class",
        "laundry service lost {n} of my clothes",
        "water cooler near {block} is leaking and the area is slippery",
        "please fix the broken window and door lock of {room}",
        "mess timing for dinner should be extended for students with late labs",
        "electricity goes off every evening in the hostel and the power backup does not work",
        "need permission for a late entry in the hostel on {date}",
        "pest control needed in {room}, there are cockroaches everywhere",
        "library book {course} reference copy is not available, need more copies",
        "gym equipment in the sports complex is broken",
        "drinking water RO in the mess is not working",
        "the lift in {block} has been stuck for {n} days",
        "bus pass for route {n} has not been issued",
        "parking space near {block} is not enough for two wheelers",
        "sanitary pad vending machine in the girls hostel is empty",
        "mattress and pillow in {room} are torn, please replace",
        "medical room in the hostel is closed at night, needed first aid",
        "short circuit smell from the switch board in {room}",
    ],
}

# Short secondary clauses used to build multi-intent (ambiguous) tickets.
SECONDARY: dict[str, list[str]] = {
    "IT_WiFi": ["also the wifi is too slow to even open the page", "and my email password is also not working",
                "maybe it is a network issue because the internet keeps dropping", "the lab system also hangs"],
    "LMS_Portal": ["and the student portal still shows it as pending", "also I cannot login to the portal to check",
                   "the LMS also shows an error", "the portal is showing {err} when I try to check status"],
    "Academics_Exams": ["and my exam is next week", "otherwise my hall ticket will be blocked",
                        "this is affecting my attendance", "my result is also withheld because of this"],
    "Fees_Accounts": ["even though the fee was paid on time", "and the payment of {amt} was deducted",
                      "and I already paid the fine", "the accounts office asked me to raise a ticket"],
    "Hostel_Facilities": ["and there is no electricity in the hostel either", "I am staying in {room}",
                          "the hostel warden asked me to write here", "also the room has no water"],
}

# Hand-written cross-department tickets with a deliberately chosen primary label.
CROSS: list[tuple[str, str]] = [
    ("fee paid but portal shows unpaid and I cannot access my courses", "Fees_Accounts"),
    ("paid {amt} exam fee but hall ticket not showing on the exam portal", "Fees_Accounts"),
    ("hall ticket not downloading from the portal, page gives {err}", "Academics_Exams"),
    ("cannot login to portal to see my result", "LMS_Portal"),
    ("result is out but portal shows {err} when I open the marks page", "LMS_Portal"),
    ("wifi not working in the hostel room {room}", "IT_WiFi"),
    ("hostel internet down since {date}, warden says contact IT", "IT_WiFi"),
    ("library system cannot issue books, the software is down", "IT_WiFi"),
    ("library fine still showing on portal after payment", "Fees_Accounts"),
    ("attendance on the portal is not updated for {course}", "Academics_Exams"),
    ("attendance section of the portal app crashes", "LMS_Portal"),
    ("scholarship status not visible on the student portal", "Fees_Accounts"),
    ("hostel room allotment list is not visible on the portal", "Hostel_Facilities"),
    ("online exam on the LMS for {course} closed before I could submit due to wifi", "Academics_Exams"),
    ("bus fee paid but bus pass not issued", "Hostel_Facilities"),
    ("lab exam of {course} cannot happen because the systems in {lab} are down", "IT_WiFi"),
    ("password for the fee payment portal is not working", "Fees_Accounts"),
    ("re-evaluation fee deducted twice, transaction {txn}", "Fees_Accounts"),
    ("timetable on LMS is different from the one shared by the department", "Academics_Exams"),
    ("electricity cut in {block} so the lab computers are off", "Hostel_Facilities"),
]

# Very short / vague tickets with only weak cues.
VAGUE: dict[str, list[str]] = {
    "IT_WiFi": ["net not working", "wifi issue pls help", "system problem in lab", "password issue", "cant connect"],
    "LMS_Portal": ["portal not opening", "lms problem", "cant submit", "login issue", "page error {err}"],
    "Academics_Exams": ["exam query", "result issue", "attendance problem", "marks wrong", "hall ticket?"],
    "Fees_Accounts": ["payment issue", "fee problem", "refund?", "money deducted", "receipt pls"],
    "Hostel_Facilities": ["room problem", "mess issue", "no water", "bus late again", "fan not working"],
}

GREETINGS = ["", "", "", "Hi, ", "Hello, ", "Dear Sir, ", "Respected Madam, ", "Hello team, ", "Good morning, ",
             "Sir, ", "To the helpdesk, "]
DETAILS = ["", "", "", "", "My student ID is {sid}.", "Roll no: {sid}.", "This is {sid} from {sem} semester.",
           "I have tried multiple times since {date}.", "Contact me at {phone}.", "You can mail me at {email}.",
           "I already raised ticket {ticket} but no response.", "Screenshot attached.", "It happened at {time}.",
           "Please treat this as urgent.", "My exam is tomorrow!", "This is the second time I am reporting this.",
           "I am a {sem} semester student.", "Refer {url}"]
REQUESTS = ["", "", "Please look into it.", "Kindly resolve asap.", "pls help", "Please do the needful.",
            "Kindly help.", "Who should I contact?", "Need this fixed urgently.", "Thanks.", "Please reply soon.",
            "Waiting for your response."]
SIGNOFFS = ["", "", "", "", "Regards, {person}", "Thanks, {person}", "- {person}"]

ABBREV = {"please": "pls", "account": "acct", "because": "bcoz", "you": "u", "are": "r", "regarding": "regd",
          "examination": "exam", "message": "msg", "information": "info", "semester": "sem", "wi-fi": "wifi",
          "laptop": "lappy", "thanks": "thx", "not": "nt", "cannot": "cant", "with": "wid",
          "assignment": "assgnmt", "department": "dept", "receipt": "reciept"}
EMOJIS = ["🙏", "😡", "😭", "😢", "🙁", "!!", "😤", "👍"]

ALT_RE = re.compile(r"\[([^\[\]]*)\]")


class TicketGenerator:
    """Builds tickets from templates; all randomness comes from one seeded RNG."""

    def __init__(self, seed: int = 42):
        self.r = random.Random(seed)

    # ------------------------------------------------------------------ templating
    def fill(self, template: str) -> str:
        """Resolve [a|b] alternatives, then {slot} placeholders."""
        template = ALT_RE.sub(lambda m: self.r.choice(m.group(1).split("|")), template)
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        return template.format(**{f: SLOTS[f](self.r) for f in fields})

    def compose(self, core: str) -> str:
        parts = [self.fill(self.r.choice(GREETINGS)) + self.fill(core)]
        sentence_end = self.r.choice([".", ".", ".", "", "?", "!"])
        parts[0] += sentence_end
        for bank, p in ((DETAILS, 0.8), (REQUESTS, 0.8), (SIGNOFFS, 0.6)):
            if self.r.random() < p:
                piece = self.fill(self.r.choice(bank))
                if piece:
                    parts.append(piece)
        text = " ".join(parts)
        return text[0].upper() + text[1:] if text and self.r.random() < 0.7 else text

    # ------------------------------------------------------------------ noise
    def _typo(self, word: str) -> str:
        if len(word) < 4 or not word.isalpha():
            return word
        i = self.r.randint(1, len(word) - 2)
        op = self.r.choice(["swap", "drop", "dup", "near"])
        if op == "swap":
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if op == "drop":
            return word[:i] + word[i + 1:]
        if op == "dup":
            return word[:i] + word[i] + word[i:]
        return word[:i] + self.r.choice("aeioustnr") + word[i + 1:]

    def add_noise(self, text: str) -> str:
        r = self.r
        if r.random() < 0.55:  # ~55% of tickets get some noise
            words = text.split(" ")
            out = []
            for w in words:
                lw = w.lower().strip(".,!?")
                if lw in ABBREV and r.random() < 0.45:
                    w = w.lower().replace(lw, ABBREV[lw])
                elif r.random() < 0.05:
                    w = self._typo(w)
                out.append(w)
            text = " ".join(out)
            if r.random() < 0.12:
                text = re.sub(r"\b(please|pls|help|sir)\b", lambda m: m.group(1) + m.group(1)[-1] * r.randint(2, 4), text)
            if r.random() < 0.15:
                text = re.sub(r"[.?!]$", r.choice(["!!!", "???", "!!", "...."]), text) or text
            if r.random() < 0.10:
                text = text.translate(str.maketrans("", "", ".,"))
        case = r.random()
        if case < 0.30:
            text = text.lower()
        elif case < 0.33:
            text = text.upper()
        if r.random() < 0.06:
            text += " " + r.choice(EMOJIS)
        return re.sub(r"\s{2,}", " ", text).strip()

    # ------------------------------------------------------------------ ticket kinds
    # Each method also returns a template_id so clean_split.py can hold out whole templates
    # (an "unseen phrasing" test set that measures generalisation beyond the templates).
    def standard(self, label: str) -> tuple[str, str]:
        k = self.r.randrange(len(CORE[label]))
        return self.add_noise(self.compose(CORE[label][k])), f"core:{label}:{k}"

    def ambiguous(self, label: str, labels: list[str]) -> tuple[str, str, str]:
        if self.r.random() < 0.4:
            k = self.r.randrange(len(CROSS))
            tpl, lab = CROSS[k]
            return self.add_noise(self.compose(tpl)), lab, f"cross:{k}"
        other = self.r.choice([l for l in labels if l != label])
        k = self.r.randrange(len(CORE[label]))
        core = CORE[label][k] + self.r.choice([", ", " but ", " and ", ". "]) + self.r.choice(SECONDARY[other])
        return self.add_noise(self.compose(core)), label, f"core:{label}:{k}"

    def vague(self, label: str) -> tuple[str, str]:
        k = self.r.randrange(len(VAGUE[label]))
        return self.add_noise(self.fill(VAGUE[label][k])), f"vague:{label}:{k}"


def random_created_date(r: random.Random, label: str) -> str:
    """Dates Jul-Nov 2026 with simple seasonality (fees spike Jul/Aug, exams spike Nov)."""
    start = date(2026, 7, 1)
    day = r.randint(0, 152)
    if label == "Fees_Accounts" and r.random() < 0.4:
        day = r.randint(0, 50)
    if label == "Academics_Exams" and r.random() < 0.4:
        day = r.randint(110, 152)
    return (start + timedelta(days=day)).isoformat()


def generate(cfg: dict, n: int | None = None) -> pd.DataFrame:
    dcfg = cfg["data"]
    labels: list[str] = cfg["labels"]
    n = n or dcfg["n_tickets"]
    gen = TicketGenerator(cfg["seed"])
    r = gen.r
    weights = [dcfg["class_weights"][l] for l in labels]
    sources = ["web_form", "email", "mobile_app", "walk_in"]

    rows = []
    for i in range(n):
        label = r.choices(labels, weights)[0]
        u = r.random()
        if u < dcfg["ambiguous_ratio"]:
            text, label, tid = gen.ambiguous(label, labels)
            kind = "ambiguous"
        elif u < dcfg["ambiguous_ratio"] + dcfg["vague_ratio"]:
            (text, tid), kind = gen.vague(label), "vague"
        else:
            (text, tid), kind = gen.standard(label), "standard"
        if r.random() < dcfg["label_noise"]:  # simulate a mis-filed ticket
            label = r.choice([l for l in labels if l != label])
            kind += "+mislabel"
        rows.append({"ticket_id": f"TKT{100000 + i}", "text": text, "category": label,
                     "source": r.choices(sources, [0.45, 0.30, 0.20, 0.05])[0],
                     "created_date": random_created_date(r, label), "gen_kind": kind, "template_id": tid})

    df = pd.DataFrame(rows)
    # inject exact duplicates and empty rows (real exports contain both)
    dups = df.sample(dcfg["n_duplicates"], random_state=cfg["seed"]).copy()
    empties = df.sample(dcfg["n_empty"], random_state=cfg["seed"] + 1).copy()
    empties["text"] = [None if k % 2 else "   " for k in range(len(empties))]
    df = pd.concat([df, dups, empties], ignore_index=True).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None, help="number of tickets (default from config)")
    ap.add_argument("--out", default=None, help="output CSV path (default from config)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    df = generate(cfg, args.n)
    out = resolve(args.out or cfg["paths"]["raw_data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("Wrote %d rows -> %s", len(df), out)
    log.info("Class counts:\n%s", df["category"].value_counts().to_string())
    log.info("Ticket kinds:\n%s", df["gen_kind"].value_counts().to_string())
    for _, row in df.dropna(subset=["text"]).sample(8, random_state=1).iterrows():
        log.info("[%s] %s", row["category"], row["text"])


if __name__ == "__main__":
    main()
