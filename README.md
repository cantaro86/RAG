# RAG for PDF


##### Debugging

```bash
module load conda
salloc --job-name="rag" --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --time=08:45:00 --nodelist=dgx01 --qos=mira
conda activate RAG
python -m debugpy --listen 0.0.0.0:5643 --wait-for-client rag_pdf_langchain.py
```

In another terminal

```bash
ssh -N -L 5643:dgx01:5643 dgx01
```
