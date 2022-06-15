FROM python:3.8

# set a directory for the app
WORKDIR .

# copy all the files to the container
COPY . .

# install dependencies
#git checkout -b x-stance
RUN pip install -e ".[dev]"

# define the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python /main.py --model gpt2 --model_args device=cuda:0 --tasks x_stance --num_fewshot 2"]
	