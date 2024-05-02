#Base image
FROM pytorch/pytorch

#Working directory inside the containerls 
WORKDIR /app1

# Copy the project directory into the container
COPY . .

#Install dependencies
RUN pip3 install -r requirements.txt

#Run the code
CMD ["python3", "mahalanobis_calculation.py"]
