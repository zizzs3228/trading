import schedule
import time

def job():
    print("Code running at ",time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

schedule.every(3).seconds.do(job)

while True:
    schedule.run_pending()