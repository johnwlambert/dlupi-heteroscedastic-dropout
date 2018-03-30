import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
fig= plt.figure(dpi=200, facecolor='white')
plt.style.use('ggplot')
sns.set_style({'font.family': 'Times New Roman'})
no_xstar_fixed_decay = { #32 : 0.1563,
				40 : 0.1918,
				50 : 0.23954,
				60 : 0.26924,
				75 : 0.3123,
				137 : 0.42326,
				200: 0.4884,
				400 : 0.5969,
				600 : 0.6335 }
no_xstar_adaptive_decay = { 40 :	0.2444,
								50 :	0.28302,
								60 :	0.33128,
								75 :	0.3729,
								137 :	0.4961,
								200 :	0.55993,
								600 :	0.6667 }
xstar = {
		40: 0.2815,
		60: 0.3531,
		75: 0.4226,
		137: 0.49764,
		200: 0.5551,
		600: 0.6668 
	}


palette = np.array(sns.color_palette("hls", 3))
plt.plot(np.log(1000.0*np.array(list(no_xstar_fixed_decay.keys()))), np.log(np.array(list(no_xstar_fixed_decay.values()))),label='no $x^{\star}$ w/ fixed l.r. decay', color=palette[0], linewidth=3.0)
plt.plot(np.log(1000.0*np.array(list(no_xstar_adaptive_decay.keys()))), np.log(np.array(list(no_xstar_adaptive_decay.values()))),label='no $x^{\star}$ w/ adaptive l.r. decay', color=palette[1], linewidth=4.0)
plt.plot(np.log(1000.0*np.array(list(xstar.keys()))), np.log(np.array(list(xstar.values()))), label='$x^{\star}$', color=palette[2], linewidth=3.0)
plt.xlabel("log(training set size)" )
plt.ylabel("log(top-1 accuracy)" )
plt.legend(loc='lower right')
fig.tight_layout(pad=4)
plt.show() #savefig('fig.pdf')
