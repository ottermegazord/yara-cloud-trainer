
'''
***************************************************************************************
*
*                   Yara Cloud Segmentation
*
*
*  Name : Idaly Ali
*
*  Designation : Data Scientist
*
*  Description : Sample SkyWeather
*
*
***************************************************************************************

'''
########################

""" Import Libraries """

import skyweatherCloud

########################

""" Parameters """

file_name = "images/img_14.jpg"
model_file = "model/yaraCloudNet_v1.pb"
label_file = "model/yaraCloudNet_v1.txt"


""" Create Skyweather Cloud Object"""

cloud = skyweatherCloud.Cloud(file_name, model_file, label_file)

""" Print cloud classification and % """

print(cloud.pred())