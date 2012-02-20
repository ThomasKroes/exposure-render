import shutil

shutil.copyfile("C:/Workspaces/ExposureRender/Build/RelWithDebInfo/ErCore.dll", "C:/Workspaces/Voreen/voreen/ext/er/ErCore.dll")
shutil.copyfile("C:/Workspaces/ExposureRender/Build/RelWithDebInfo/ErCore.lib", "C:/Workspaces/Voreen/voreen/ext/er/ErCore.lib")
shutil.copyfile("C:/Workspaces/ExposureRender/Source/Core.cuh", "C:/Workspaces/Voreen/voreen/ext/er/Core.cuh")
shutil.copyfile("C:/Workspaces/ExposureRender/Source/General.cuh", "C:/Workspaces/Voreen/voreen/ext/er/General.cuh")

print "Files copied..."
