from kuka_env import KukaEnv
import numpy as np


GUI = True
make_gif = False
env = KukaEnv(GUI=GUI)

# plot trajectory
path = [(0.3542580627450649, -2.022089521226548, -2.2182339069315398, 0.9115339602971932, 2.752754863673904, 0.36717590416693646, 1.448471191228442),
        (0.4271312500221772, -0.7784759627038191, -1.0505875813343133, 1.355223296431746, 0.23683135663415855, 0.12325083494663351, -0.7197452102854228),
        (0.5881303831253089, 0.5849310274380142, -0.05696028570145195, 1.6483631621211003, -1.8229752960204042, -0.022967923170421845, -2.503331871691698)]
gifs = env.plot(path, make_gif=make_gif)
if make_gif:
    from PIL import Image

    # Setup the 4 dimensional array
    a_frames = []
    for im_frame in gifs:
        a_frames.append(np.asarray(im_frame))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims[0].save("kuka.gif", save_all=True, append_images=ims[1:], loop=0, duration=10)
