def calculate_iou_v2(y_true, y_pred):
    conf_threshold = 0.5
    y_pred_decoded = decode_y(y_pred,
                              confidence_thresh=0.01,
                              iou_threshold=0.45,
                              top_k=100,
                              input_coords='centroids',
                              normalize_coords=True,
                              img_height=img_height,
                              img_width=img_width)
    y_true_decoded = decode_y(y_true,
                              confidence_thresh=0.01,
                              iou_threshold=0.45,
                              top_k=100,
                              input_coords='centroids',
                              normalize_coords=True,
                              img_height=img_height,
                              img_width=img_width)