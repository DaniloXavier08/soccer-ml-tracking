import cv2

# Carregar imagem
img = cv2.imread('img/frame_liverpool.png')

# Criar detector ORB
orb = cv2.ORB_create()

# Detectar keypoints e descritores
keypoints, descriptors = orb.detectAndCompute(img, None)

# Desenhar keypoints
img_com_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

cv2.imshow("Keypoints", img_com_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
