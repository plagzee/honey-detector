# This is a file to download images for the dataset. 
# Downloaded Content as for 08-06-2025 01:01PM IST
# 50 dog images
# 50 cat images
# 35 person images
# 35 tree images
# 30 furniture images
# 30 bottle images
# 30 laptop images
# 30 table images
# 30 bedroom images
# 30 street images
# 20 car images




from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

# key_words = [
#     "dog",
#     "cat",
#     "person",
#     "tree",
#     "furniture",
#     "bottle",
#     "laptop",
#     "table",
#     "bedroom",
#     "street",
#     "car",
# ]

# for keyword in key_words:
#     response().download(keyword, 35)

# remaining_keywords = [
#     "furniture",
#     "bottle",
#     "laptop",
#     "street",
#     "car",
# ]

last_kw = "car"


# for kw in remaining_keywords:
#     response().download(kw, 30)
#     print("Downloaded: ", kw)

print("Downloading: ", last_kw)

response().download(last_kw, 30)

print("Required Images Downloaded Successfully")    