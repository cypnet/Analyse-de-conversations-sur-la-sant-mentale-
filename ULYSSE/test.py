import pandas as p
import matplotlib as plt

#Recuperation des donnees
csvF = p.read_csv("./train.csv");
context = csvF["Context"]
response = csvF["Response"]

#Donnee sans doublons
contextD= p.DataFrame(context)
contextDistinct = contextD.drop_duplicates(contextD)

reponseD= p.DataFrame(response)
reponseDistinct = reponseD.drop_duplicates(reponseD)

print(len(contextDistinct))
print(len(reponseDistinct))

