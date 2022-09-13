FROM node:18-alpine

RUN npm install -g http-server

WORKDIR /app

COPY package*.json ./

# Install dependencies
RUN npm install

# Copy files
COPY . .

# Build
RUN npm run build

EXPOSE 8080
CMD [ "http-server", "dist" ]