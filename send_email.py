import smtplib
import getpass
from email.mime.text import MIMEText


# defining a function for sending email
def send_email():
    senders_address = 'ankurrai800@gmail.com'
    password = getpass.getpass()            # getpass will not show password, written on terminal
    subject = 'Learn.Inspire.Grow.2'
    message = '''
        Hello Everyone!
        Enjoying the AI Mafia course!
        
        Thank you!
        Ankur Kumar Rai 
    '''
    recipients = ['ankurrai800@gmail.com', 'example@gmail.com']    # To

    # Server Initialization
    server = smtplib.SMTP('smtp.gmail.com', 587)    # enables SMTP server
    server.starttls()
    server.login(senders_address, password)

    # draft my message body
    message = MIMEText(message)
    message['Subject'] = subject
    message['From'] = senders_address
    message['To'] = ', '.join(recipients)
    message.set_param('importance', 'high value')

    server.sendmail(senders_address, recipients, message.as_string())


send_email()