import numpy as np
import matplotlib.pyplot as plt

r = np.genfromtxt('rad30.dat')
a = r.reshape((30,30))

plt.imshow(a)
plt.colorbar()
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title('Single-scattered transmittance')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(1,20,'plume shadow',bbox=props,fontsize=14)
plt.text(7,2,'plume',bbox=props,fontsize=14)
plt.text(17,28,'wind direction',bbox=props,fontsize=14)
plt.arrow(26,26,-5,-5,width=0.4)
plt.text(18,4,'incident solar\nradiation',bbox=props,fontsize=14)
plt.arrow(28,5,-5*1.17,5*0.79,width=0.4,color='y')
plt.savefig('convg_ss_30x30.pdf')
plt.show()
