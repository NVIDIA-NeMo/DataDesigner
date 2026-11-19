Repair `clean.py`. Its `dedupe(rows)` function takes a list of dictionaries
from `orders.csv`. Return the original record with the newest ISO date in
`updated_at` for each `order_id`, preserving first-seen order across distinct
IDs. On equal dates keep the first record. Keep the function signature, do not
mutate the input, and edit only `clean.py`.

Run `python public_test.py` to reproduce the failure and check your fix.
