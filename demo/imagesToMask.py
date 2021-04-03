images= glob.glob('./test_images/*')
for idx, image in enumerate(images):
    img= cv2.imread(image)
    outputs= predictor(img)
    mask= outputs['instances'].get('pred_masks')
    mask= mask.to('cpu')
    num, h, w= mask.shape
    bin_mask= np.zeros((h, w))
    
    for m in mask:
        bin_mask+= m
    filename= './bin_masks/'+str(idx+1)+'.png'
    cv2.imwrite(filename, bin_mask)
