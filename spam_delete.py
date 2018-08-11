import imaplib

mail = imaplib.IMAP4_SSL('imap.gmail.com')
mail.login('YOUR_EMAIL', 'PASSWORD')
mail.select('"[Gmail]/Spam"')
typ, data = mail.search(None, 'ALL')
spams = len(data[0].split())
for num in data[0].split():
    ty, dat = mail.store(num, '+FLAGS', '\\Deleted')
mail.expunge()
mail.close()
mail.logout()
print('{} spams deleted.'.format(spams))
