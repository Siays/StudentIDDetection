import re
from datetime import datetime

def check_id(string):
    student_id_reg = r'(?:\d|O){2}[A-Z]{3}(?:\d|O){5}' # digit | o in uppercase , accept o in between the digits
    match = re.search(student_id_reg, string)

    if match:
        print("matched")
        return True

    return False


def check_exp_date(string):
    exp_date_reg = r'\b(?:0[1-9]|[12][0-9]|3[01])-(?:0[1-9]|1[0-2])-(?:20)\d{2}\b'  # Pattern to match expiry date format

    # Check if the input string matches the pattern
    match = re.search(exp_date_reg, string, flags=re.IGNORECASE)
    if match:
        # Extract the expiry date
        expiry_date_str = match.group()

        print(expiry_date_str)

        # Convert expiry date string to datetime object
        try:
            expiry_date = datetime.strptime(expiry_date_str, "%d-%m-%Y")
        except ValueError:
            print("Invalid expiry date format")
            return False

        # Get current date
        current_date = datetime.now()

        # Compare expiry date with current date
        if expiry_date > current_date:
            print("Not expired.")
            return True
        else:
            print("Expired.")

    return False

