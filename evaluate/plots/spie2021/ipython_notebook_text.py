import os
import yaml
from evaluate import make_boxplots, convert_to_df

result_filename = "/home/jorg/expers/cardiac_sr/acdc128/results_non_centered_2907202.yml"
with open(result_filename, 'r') as fd:
    ttest_results = yaml.load(fd)
fd.close()
del ttest_results['ACAI']
del ttest_results['AE+C-loss+']
del ttest_results['AE+C-loss']
del ttest_results['AE']
ttest_results['ae'] = ttest_results['AE-mse']
del ttest_results['AE-mse']
print(ttest_results.keys())
df = convert_to_df(ttest_results)


fig_name = "boxplot_proposed_versus_coventional_synth_only.png"
fig_name = os.path.join(os.path.join("/home/jorg/expers/cardiac_sr/acdc128/", "results"), fig_name)
synth_only = False
do_save = True
method_key_filter = ['ae', 'linear', 'bspline', 'lanczos']
make_boxplots(ttest_results, fig_name, do_show=True, do_save=do_save, use_fill_color=False,
              show_means=False, df=df, method_key_filter=method_key_filter,
              synth_only=synth_only)



