import anseri as ai
import os
import email
import re

emailRegex = re.compile(r'''(
[a-zA-Z0-9._%+-]+
@
[a-zA-Z0-9.-]+
(\.[a-zA-Z]{2,4})
)''', re.VERBOSE)


def email_parsing(s):
    b = email.message_from_string(s)
    frm=[]
    to=[]
    if b['from']:
        for groups in emailRegex.findall(b['from']):
            frm.append(groups[0])
    if b['to']:
        for groups in emailRegex.findall(b['to']):
            to.append(groups[0])

    return({'body': b.get_payload(), 'author': list(set(frm)), 'to': list(set(to)), 'date': '14 Nov 2016'})

def create_anseri_db(source, dbname):
    # source is an iterable collection of dicts (see main for example)
    dbpath='/home/gabriel/.ai/data/' + dbname + '.db'
    contentcols = ["body", "author", "to"]
    # timecol = 'date'
    # timecol = 'date'
    # print(source, dbname, dbpath, contentcols, timecol)
    ai.data.backends.sqlite.SQLiteController_FTS.from_iterable(source, dbpath, contentcols)


def testing_some_anseri_fns(dbname):
    d = ai.Dataset(dbname)
    selection = ai.AllSelection()
    print(d.time_range)
    print([ai.utc.mth(x) for x in d.time_range])
    selection = ai.TimeSelection(('Mar 1 2012', 'Jan 2 2013'))
    selection = ai.FullTextSelection("Alice")
    print(len(selection.docids))
    model = d.load(selection)
    print(selection.docids)
    print(model.shape)

def take_enron_emails():
    source = []
    os.chdir('/home/gabriel/xberkeley/ml/juris/')
    dire = 'enron_mail_20150507/maildir'
    for person in os.scandir(dire):
        if not person.name.startswith('.') and person.is_dir():
            try:
                for mail in os.scandir(dire + '/' + person.name + '/inbox'):
                    if not mail.name.startswith('.') and mail.is_file():
                        f = open(dire + '/' + person.name + '/inbox/' + mail.name, 'r')
                        try:
                            source.append(f.read())
                        except UnicodeDecodeError:
                            pass
            except FileNotFoundError:
                print('error with', person.name)
    return(source)

if __name__ == '__main__':
    new_source = []
    source = take_enron_emails()
    for eml in source:
        new_source.append(email_parsing(eml))
    dbname = 'enron'
    create_anseri_db(new_source, dbname)
    testing_some_anseri_fns(dbname)

    
        