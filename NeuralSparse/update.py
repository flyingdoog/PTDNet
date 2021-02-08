import mysql.connector
mydb = mysql.connector.connect(
  host="104.39.162.41", #"",
  user="dul262",
  passwd="dul262dgx1"
)

mycursor = mydb.cursor()


mycursor.execute("SELECT id FROM dgx1.GCN")

myresult = mycursor.fetchall()
ids = []
ops = []

for id in myresult:
    id = id[0]
    # sql = 'UPDATE dgx1.results SET id = \'' + id+ '-0\' WHERE id = \''+id+'\'';
    newid = ''
    sp = id.split('-')
    if len(sp)<11:
        continue
    for i in range(1,len(sp)):
        newid+=sp[i]+'-'
    newid = newid[:-1]
    print(newid)

    sql = 'UPDATE dgx1.GCN SET id = \'' + newid+ '\' WHERE id = \''+id+'\'';

    mycursor.execute(sql)
    mydb.commit()