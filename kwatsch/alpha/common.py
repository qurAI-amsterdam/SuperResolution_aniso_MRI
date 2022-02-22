

def get_alpha_probe_features(feature_dict, batch_item):
    feature_dict['spacing'] = batch_item['spacing'][0]
    feature_dict['norm_frame_id'] = (batch_item['frame_id'] + 1) / batch_item['orig_num_frames']
    feature_dict['orig_num_slices'] = batch_item['image'].shape[0]
    return feature_dict
