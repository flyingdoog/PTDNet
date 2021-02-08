import mysql.connector
mydb = mysql.connector.connect(
  host="104.39.162.41", #"",
  user="dul262",
  passwd="dul262dgx1"
)

mycursor = mydb.cursor()
# sql = 'INSERT INTO dgx1.sparseGCN VALUES (\'test4\',0.5)'
# mycursor.execute(sql)
# mydb.commit()


mycursor.execute("SELECT * FROM dgx1.sparseGCN")

myresult = mycursor.fetchall()
print(myresult)
for x in myresult:
  print(x)