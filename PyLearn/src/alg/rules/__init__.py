import ConfigParser

cf = ConfigParser.ConfigParser()
cf.read("..\\resource\\config")

vegetable = cf.get("Food",  "Vegetable")