import shutil

shutil.copyfile("C:/Workspaces/ExposureRender/Build/RelWithDebInfo/ErCore.dll", "C:/Workspaces/Voreen/voreen/modules/er/ext/er/bin/ErCore.dll")
shutil.copyfile("C:/Workspaces/ExposureRender/Build/RelWithDebInfo/ErCore.lib", "C:/Workspaces/Voreen/voreen/modules/er/ext/er/lib/ErCore.lib")
shutil.copyfile("C:/Workspaces/ExposureRender/Source/Core.cuh", "C:/Workspaces/Voreen/voreen/modules/er/ext/er/include/Core.cuh")
shutil.copyfile("C:/Workspaces/ExposureRender/Source/General.cuh", "C:/Workspaces/Voreen/voreen/modules/er/ext/er/include/General.cuh")

print "Files copied..."
