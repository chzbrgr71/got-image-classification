FROM node:11.15-alpine

WORKDIR /usr/src/app
COPY package*.json ./
RUN npm ci

COPY . .
ENV NODE_ENV "development"
EXPOSE 3000

CMD [ "npm", "run", "container" ]