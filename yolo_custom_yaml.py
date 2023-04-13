# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path:   # dataset root dir
train: C:\allmodel\yolov5\coco128\datasets/train/images/  # train images (relative to 'path') 128 images
val: C:\allmodel\yolov5\coco128\datasets/valid\images/ # val images (relative to 'path') 128 images
test: C:\allmodel\yolov5\coco128\datasets/test\images/ # test images (optional)


nc : 32
# # Classes
# names: ['scratch']
names: [ "Gucci","Gucci_tudor","GATE2","Valet_lounge",
"BottegaVeneta","Fendi","Bulgari","Omega","Charger","Elevator_to_charger",
"Dior","Tods","Tudor","Balenciaga","Thombrowne_women",
"Moncler","Boucheron","Saint_laurent","Valentino", "Tiffany", 
"Iwc","Jaegerlecoultre","Gucci_beauty","Breitling","Girard_perregaux",
"Hublot","Prada","Burberry","Panerai","montblanc","Zenith","TAGheuer"]



# Download script/URL (optional)
download: https://ultralytics.com/assets/coco128.zip
