# Self_Driving_Car

## Installing requirements

## Requires models directory, and data directory (if you plan to train any models)

## To run the most basic demo:

# 1. Move both .pt files in the outermost directory into the /models directory.

# 2. Enter the /GTA directory 'cd /path/to/Self_Driving_Car/GTA'

# 3. Ensure the 'DRIVE' flag on line 8 of drive.py is set to False.

# 4. Play media in the top left of your screen (a youtube video, for instance) at 800 x 600 pixels.

When the script starts, it will open a window that plays back what the segmentation model is seeing. This can be used to adjust the video into place. Terminal output represents the predicted control.

# CARLA

This is if you want to gather training data or see the model in action in the environment it was trained in. This will take some time to set up, though, and some bugs may persist in the Jupyter Notebook(it got a little messy).
To get the most up-to-date version, our repository is [here](https://github.com/henrynitzberg/Self_Driving_Car.git).

## CARLA Installation

Follow the CARLA installation from the docs. We use the current latest release of CARLA, v0.9.15. You can find it [here](https://github.com/carla-simulator/carla/blob/master/Docs/download.md)
If you want, you can also follow this video [here](https://www.youtube.com/watch?v=jIK9sanumuU). This is an installation of 0.9.14, but it remains the same. You don't need the "additional maps," though we have it, so if something goes wrong, try installing it and then running it again(but you really shouldn't need to)

Be warned, this is a very sizable installation. With additional maps, this can go up to 70+ GB, though the base installation is just 20GB.

You can open your CARLA installation by going into CARLA > WindowsNoEditor > CARLAUE4.exe (make a shortcut or something). Play around if you want!

## Environment Setup

We'll outline everything you need, but you can also follow this video [here](https://www.youtube.com/watch?v=zZ8s_qrKYGE) to set up the basic Python configuration, except when asked to make your own Conda environment, go to step 3.

1. Install Anaconda or open it if you have it. (you can also do this in terminal, if you prefer, but conda is just easier)
2. cd into the directory of this README.
3. run `conda env create -f carla_environment.yml` or you can specify an environment name with `conda env create -n custom_env_name -f carla_environment.yml`
4. if you made one by accident or want to add to an existing environment, run `conda env update --file environment.yml --prune` instead.

## Running the environment

1. You might be able to run directly in VSCode, but I have been using Jupyter Notebook, so run `jupyter notebook` in conda.
2. Once a tab opens, click into Driving > carla_control_2.ipynb
3. Run your CARLA installation.
4. Only once it loads, you can then run all the cells until 'traffic manager setup'
5. Traffic manager setup will run it in synchronous mode, which may make the CARLA window inaccessible. That's normal as a spectator hasn't been set up yet.
6. Once you've taken a breather, run traffic manager setup
7. Now SKIP until you reach the part that says "Test with UI in separate window here"
8. Running that will begin the simulation. it will open a window that displays the model's current commands.
9. You can go around with the spectator in CARLA with WASD. hit 'q' to end the simulation
