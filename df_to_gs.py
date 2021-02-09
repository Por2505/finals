import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# define the scope
scope = ['https://www.googleapis.com/auth/drive']

# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)

# authorize the clientsheet 
client = gspread.authorize(creds)

# get the instance of the Spreadsheet
sheet = client.open('Class Attendance')

# get the first sheet of the Spreadsheet
worksheet = sheet.get_worksheet(0)
worksheet.update_cell(2,4,'1')
results = worksheet.get_all_records()
print(results)