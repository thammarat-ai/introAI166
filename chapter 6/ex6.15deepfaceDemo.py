#Example 6.15
from deepface import DeepFace

#Detect Faces ==========================================================
detected_face = DeepFace.extract_faces("oldwoman.jpeg")


#Verify images==========================================================
result = DeepFace.verify("face11a.jpeg", "face11a.jpeg")
print("Is verified: ", result["verified"])

# # Get age, gender, race, emotion =======================================
# #from deepface import DeepFace
demography = DeepFace.analyze("perry1.jpg")


print("Age: ", demography[0]['age'])
print("Emotion: ", demography[0]['dominant_emotion'])
print("Gender: ", demography[0]['dominant_gender'])
print("Race: ", demography[0]['dominant_race'])
