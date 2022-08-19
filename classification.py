#given data is not directly usefull for training the model
#classification.py crops the faces from the given data to use for training the model
#it saves the cropped faces in output foler in  categories
    #dataset_helmet   - with_helmet / without_helmet
    #dataset_mask     - with_mask / without_mask

import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt   
from tqdm import tqdm

path = "dataset\\annotations.xml"

tree = ET.parse(path)
root = tree.getroot()

has_masks_count = 0
has_helmet_count = 0
no_masks_count = 0
no_helmet_count = 0
mask_invisible_count = 0
has_both = 0

#set pixel value to avoid less quality faces
crop_value = 40

saveDir = "output\\"
pathToImages = "dataset\images\\"


for image in tqdm(root):
    #print ("...........image...........")
    if len(image.attrib):
        #read the image
        imageId =  image.attrib["id"]
        imagePath = pathToImages + str(imageId) +".jpg"
        img = cv2.imread(imagePath)
        head_id = 0
        for heads in image:
            
            if heads.attrib["label"] == "head":
                head_id+=1
                #print (".........head.........")
                #print (heads.attrib)
                xtl, ytl, xbr, ybr = float(heads.attrib["xtl"]),float(heads.attrib["ytl"]),\
                                            float(heads.attrib["xbr"]),float(heads.attrib["ybr"])
                #print (xtl, ytl, xbr, ybr)
                start = (int(xtl), int(ytl))
                end = (int(xbr), int(ybr))
                
                #print (start,end)
                #img = cv2.rectangle(img,start,end,(0,255,0),2)
                hasmask = "no"
                hashelmet = "no"
                
                x,y = xbr-xtl,ybr-ytl
                #ignore images with bad pixel values
                if x <crop_value or y<crop_value:
                    continue
                for each_head in heads:
                    #draw rectangles on the image
                    cropped = img[int(ytl):int(ybr),int(xtl):int(xbr)]   
                    outpath = 0
                    if each_head.attrib["name"] == "has_safety_helmet":
                        hashelmet = each_head.text
                        if hashelmet =="yes":
                            has_helmet_count+=1
                            outpath =  saveDir+  "dataset_helmet\\with_helmet\\" +  str(imageId)+"-"+str(head_id)+".jpg"
                            
                            #plt.imshow(cropped)
                            #plt.show()
                        elif hashelmet =="no":
                            outpath =  saveDir+  "dataset_helmet\\without_helmet\\" +  str(imageId)+"-"+str(head_id)+".jpg"
                            
                            no_helmet_count+=1
                    
                        #print ("has_safety_helmet",hashelmet)
                    if each_head.attrib["name"] == "mask":
                        hasmask = each_head.text
                        if hasmask == "yes":
                            outpath =  saveDir+  "dataset_mask\\with_mask\\" +  str(imageId)+"-"+str(head_id)+".jpg"
                            #print (outpath)
                            has_masks_count+=1
                        elif hasmask == "no":
                            outpath =  saveDir+  "dataset_mask\\without_mask\\" +  str(imageId)+"-"+str(head_id)+".jpg"
                            
                            no_masks_count +=1
                        elif hasmask == "invisible":
                            mask_invisible_count +=1
                        #print ("mask",hasmask)
                    if outpath:
                        #print (outpath)
                        cropped = cv2.resize(cropped,(224,224))
                        cv2.imwrite(outpath,cropped)
                    if hasmask =="yes" and hashelmet == "yes":
                        has_both += 1
                    if hasmask =="no" and hashelmet == "no":
                        has_both += 1
                    
                    #print ("------------")
                    #print (each_head.attrib)
                    #print (each_head.text)
                    #break
        
        # plt.imshow(img)
        # plt.show()
        
print ("has mask count",has_masks_count)
print ("no mask count",no_masks_count)
print ("mask invisible count",mask_invisible_count)

print ("has helmet count",has_helmet_count)
print ("no helmet count",no_helmet_count)