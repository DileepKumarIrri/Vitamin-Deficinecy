import os
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

data_dir = 'dataset/train'
print(data_dir)


classes = []
for file in os.listdir(data_dir):
    if file != "clear":
        classes.append(file)



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/accuracy")
def accuracy():
    return render_template("accuracy.html")


@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("images",filename)

@app.route("/upload",methods=["POST","GET"])
def upload():
    if request.method=='POST':
        print("hdgkj")
        m = int(request.form["alg"])
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join("images/", fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)


        if m == 1:
            print("bv1")
            new_model = load_model('model/CNN.h5')
            test_image = image.load_img(mypath, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
        elif m == 2:
            print("bv2")
            new_model = load_model('model/Mobilenet.h5')
            test_image = image.load_img(mypath, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
        else:
            print("bv3")
            new_model = load_model('model/dec.h5')
            test_image = image.load_img(mypath, target_size=(128, 128))
            test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        preds = classes[np.argmax(result)]



        if preds == "aloperia areata":
            msg = "VITAMIN DEFICIENCY-D"
            msg1 =      "Consider the Mediterranean diet, which is high in fruits, vegetables, nuts, whole grains, fish and healthy oils. Take the right supplements. According to a 2018 study published in Dermatology and Therapy, you need key nutrients in your diet to prevent hair loss."

        elif preds == "beaus lines":
            msg = "VITAMIN DEFICIENCY-C"
            msg1 =      "Dark green leafy vegetables, as well as quinoa, almonds, cashews, peanuts, edamame and black beans, are good sources, too. Summary Adequate magnesium intake is crucial to prevent vertical ridges in your nails. This mineral also helps with protein synthesis and the formation of new nails"

        elif preds == "bluish nail":
            msg = "VITAMIN DEFICIENCY-B12"
            msg1 =      "Plenty of nutrients in food can help your nails, taking them from dry and brittle to healthy and strong. Foods that can improve your nails include fruits, lean meats, salmon, leafy greens, beans, eggs, nuts, and whole grains."

        elif preds == "bulging eyes":
            msg = "VITAMIN DEFICIENCY-A"
            msg1 =      "Eating foods high in potassium also helps counter the salt. These include.bananas,yogurt,potatoes,dried,apricots"

        elif preds == "cataracts eyes":
            msg =  "VITAMIN DEFICIENCY-D"
            msg1 =       "at Right. You can't do anything about your age or family history, but you can change your diet. Some research shows that eating foods high in antioxidants like vitamins C and E may help prevent cataracts. If you already have cataracts, it may slow their growth."

        elif preds == "clubbing":
            msg = "VITAMIN DEFICIENCY-D"
            msg1 =     "Meat, fish, eggs, beans and nuts. Aim for two portions a day, except for fish, which should be eaten twice a week (with one serving being oily fish)."

        elif preds == "crossed eyes":
            msg = "VITAMIN DEFICIENCY-B6"
            msg1 =     " Omega-3 fatty acid from cold-water fish like salmon, tuna, sardines and halibut reduce the risk of developing eye disease.Oranges and all of their citrus cousins grapefruit, tangerines, tomatoes and lemons are high in vitamin C, an antioxidant that is critical to eye health"

        elif preds == "Dariers disease":
            msg = "VITAMIN DEFICIENCY-A"
            msg1 =      "There is no food Diet only Treatment.Herpes simplex is treated with aciclovir or valaciclovir. Severe Darier disease is usually treated with oral retinoids, either acitretin or isotretinoin. Ciclosporin has been reported to be effective in a few patients."

        elif preds == "eczema":
            msg = "VITAMIN DEFICIENCY-D"
            msg1 =      "Vegetables and fruits that are high in inflammation-fighting flavonoids: Apples, broccoli, cherries, blueberries, spinach, and kale. Flavonoids have been found to help improve the overall health of a person's skin and fight problems such as inflammation (which is associated with eczema)."

        elif preds == "glucoma eyes":
            msg = "VITAMIN DEFICIENCY-B"
            msg1 =      "people who drank at least one cup of hot tea daily lowered their glaucoma risk by 74% compared to those who did not. The foundation also suggested chocolate, bananas, avocados, pumpkin seeds and black beans for their health benefits."

        elif preds =="Lindsays nails":
            msg = "VITAMIN DEFICIENCY-B12"
            msg1 =      "A healthy diet can keep the nails healthy and prevent the formation of hangnails. Hangnails can result from a protein deficiency, as well as a lack of essential vitamins including folic acid, vitamin B, vitamin C, and keratin.Kiwi,Broccoli,Bell peppers,Tomatoes."
        elif preds=="lip":
            msg = "VITAMIN DEFICIENCY-B2"
            msg1 =    "Eggs, milk, carrots, spinach, apricots.Helps boost immune system, p lips after sun or wind damage, and retain collagen.	Orange juice, strawberries, green peppers, citrus fruits, tomatoes, sweet potatoes"
        elif preds=="tounge":
            msg = "VITAMIN DEFICIENCY-B3"
            msg1 =      "Head to your refrigerator or grocery store and look for cool items like yogurt or applesauce. After eating, make sure to drink water to remove the stuck food debris that can harm your burning tongue."
        else:
            msg = "VITAMIN DEFICIENCY-D"
            msg1 = "The best way to ensure good eye health for yourself and your loved ones is to eat a diet that contains all the essential nutrients. Make sure to include fresh vegetables, fruits, farm-fresh eggs, nuts, and lean fish and meat in your diet."

        return render_template("template.html", text=preds, image_name=fn,msg=msg, msg1=msg1)
    return render_template("upload.html")

if __name__ == '__main__':
    app.run(debug=True)
