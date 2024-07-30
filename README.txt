Quando si esegue snake_game_human.py è viene generato l'errore: libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:$${ORIGIN}/dri:/usr/lib/dri, suffix _dri) libGL error: failed to load driver: iris libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:$${ORIGIN}/dri:/usr/lib/dri, suffix _dri) libGL error: failed to load driver: iris libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:$${ORIGIN}/dri:/usr/lib/dri, suffix _dri) libGL error: failed to load driver: swrast
Possibile soluzione:
#Enter the storage location of Anaconda libstdc++

cd /home/usr/anaconda3/lib/ 
mkdir backup  #Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
#Copy the c++ dynamic link library of the system here
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.19