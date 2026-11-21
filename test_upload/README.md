# Test files for the Batch Upload tab

These tickets were written by hand for testing. They are **not** part of the training data and are not generated from its templates.

| File | What it tests | What you should see |
|---|---|---|
| `helpdesk_test_tickets.csv` | The normal, clean case: 40 tickets, 8 per department, all four columns (`ticket_id`, `text`, `category`, `created_date`) | `text` is selected automatically. 40 tickets, 0 skipped. After **Run batch**: predictions, entities, an agreement score against `category`, and a weekly trend chart on the Dashboard |
| `edge_cases_test.csv` | A messy file: no `text` column (it's called `complaint`), one empty row, one duplicate, one unknown label (`Hostel`), a vague ticket, a multi-intent ticket, and the "Spurious cue" ticket | The app picks `complaint` and tells you so. Empty rows skipped: 1. Duplicate texts: 1. No unknown-label warning, because the label column is named `dept`, not `category` (rename it to `category` to see the warning) |

To produce the unquoted-comma error, open any CSV in Notepad, remove the double quotes around a ticket that contains a comma, and upload it.
