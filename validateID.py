import re
from datetime import datetime


def check_id(string):
    student_id_reg = r'(?:\d|O){2}[A-Z]{3}(?:\d|O){5}'  # digit | o in uppercase , accept o in between the digits
    match = re.search(student_id_reg, string)

    if match:
        student_id_str = match.group()
        return True, student_id_str

    return False, ""


def check_exp_date(string):
    # Pattern to match expiry date format
    exp_date_reg = r'\b(?:0[1-9]|[12][0-9]|3[01])-(?:0[1-9]|1[0-2])-(?:20)\d{2}\b'

    # Check if the input string matches the pattern
    match = re.search(exp_date_reg, string, flags=re.IGNORECASE)
    if match:
        # Extract the expiry date
        expiry_date_str = match.group()

        # Convert expiry date string to datetime object
        try:
            expiry_date = datetime.strptime(expiry_date_str, "%d-%m-%Y")
        except ValueError:
            return False, ""

        # Get current date
        current_date = datetime.now()

        # Compare expiry date with current date
        if expiry_date > current_date:
            return True, expiry_date_str

    return False, ""


def get_validated_info(valid_student_id, expiry_date):
    output_str = ("Valid Student."
                  "\n"
                  "\nStudent ID details:"
                  "\nStudent ID: {}"
                  "\nExpiry date: {}".format(valid_student_id, expiry_date))
    return output_str


# status, student_id = check_id("23WMR09454")
# print("status: {}, id: {}".format(status, student_id))
#
# expiry_status, date = check_exp_date("01-05-2024")
#
# print("expiry_status: {}, date: {}".format(expiry_status, date))
#
# print(get_validated_info("23WMR09454", "01-05-2024"))
