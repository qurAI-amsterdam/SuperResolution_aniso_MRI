import os
import matplotlib.pyplot as plt
from matplotlib import cm


def compare_long_axis_views(method_img_dict, patient_id, slice_id, slice_info, frame_id=None, fig_dir=None,
                            do_save=False, do_return=False, do_show=True, num_interpolations=None,
                            width=5, height=2., with_diff_image=False):
    assert num_interpolations is not None
    if do_save and fig_dir is None:
        raise ValueError("fig_dir parameter must be specified when saving figure.")
    result_dict = {}
    img_ref = method_img_dict['reference'][patient_id]['image'] if frame_id is None else method_img_dict['reference'][patient_id]['image'][frame_id]
    for meth_desc, image_dict in method_img_dict.items():
        if frame_id is not None:
            img1_m1 = image_dict[patient_id]['image'][frame_id]
        else:
            img1_m1 = image_dict[patient_id]['image']
        # if meth_desc in ['linear', 'bspline', 'lanczos']:
        #    img1_m1 = strip_conventional_interpolation_results(img1_m1, img_ref, num_interpolations + 1)
        spacing_m1 = image_dict[patient_id]['spacing']
        if len(spacing_m1) > 3:
            # unfortunately for reference images spacing has len 3 and for SR methods len 4
            spacing_m1 = spacing_m1[1:]
        zsize, ysize, xsize = img1_m1.shape
        extent_m1 = (0, ysize * spacing_m1[1], 0, zsize * spacing_m1[0])
        if slice_info is not None:
            img1_s_m1 = img1_m1[:, slice_id, slice_info]
        else:
            img1_s_m1 = img1_m1[:, slice_id]
        result_dict[meth_desc] = img1_s_m1
        print(meth_desc, img1_m1.shape, img1_s_m1.shape, spacing_m1)
        if do_show:
            if not with_diff_image or meth_desc == 'reference':
                fig = plt.figure(figsize=(width, height))
                plt.imshow(img1_s_m1, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
                             extent=extent_m1)
                plt.axis("off")
            elif with_diff_image and meth_desc != "reference":
                fig = plt.figure(figsize=(width, height * 2))
                ax = fig.subplots(2)
                ax[0].imshow(img1_s_m1, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
                             extent=extent_m1)
                ax[0].axis("off")
                ax1plot = ax[1].imshow(result_dict["reference"] - img1_s_m1, cmap=cm.bwr, vmin=-0.5, vmax=0.5, interpolation="nearest",
                           extent=extent_m1)
                # fig.colorbar(ax1plot, ax=ax[1], fraction=0.046, pad=0.04)
                ax[1].axis("off")
            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
        if do_save:
            if frame_id is not None:
                fig_file_name = os.path.join(fig_dir, meth_desc + "_lax_p{}_f{}_s{}_{}x.png".format(patient_id,
                                                                                            frame_id, slice_id,
                                                                                                num_interpolations))
            else:
                fig_file_name = os.path.join(fig_dir, meth_desc + "_lax_p{}_s{}_{}x.png".format(patient_id,
                                                                                                slice_id,
                                                                                                num_interpolations))
            plt.savefig(fig_file_name, bbox_inches='tight')
            print("Saved to {}".format(fig_file_name))
        if do_show:
            plt.show()
        plt.close()
    if do_return:
        return result_dict
