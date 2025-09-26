cd ..
git clone https://github.com/565353780/sdf-generate.git
git clone https://github.com/565353780/base-trainer.git
git clone https://github.com/565353780/point-cept.git

cd sdf-generate
./setup.sh

cd ../base-trainer
./setup.sh

cd ../point-cept
./setup.sh

pip install flash-attn --no-build-isolation

pip install -U craftsman jaxtyping
