'use strict';

const electron = require('electron');  
const app = electron.app;  
const BrowserWindow = electron.BrowserWindow;
let mainWindowIcon = __dirname + '/static/images/icon';
let mainWindow;

if (process.platform == 'darwin') {
  mainWindowIcon = mainWindowIcon + ".icns";
} else if (process.platform == 'win') {
  mainWindowIcon = mainWindowIcon + ".ico";
} else {
  mainWindowIcon = mainWindowIcon + ".png";
}

app.on('window-all-closed', function() {
  app.quit();
  //if(process.platform != 'darwin') {
  //  app.quit();
  //}
});

app.on('ready', function () {
  mainWindow = new BrowserWindow({
    title: 'kMindConnect',
    width: 800,
    height: 600,
    minWidth: 500,
    minHeight: 200,
    icon: mainWindowIcon,
    acceptFirstMouse: true,
    //titleBarStyle: 'hidden',
    //frame: false,
  });

  mainWindow.setMenu(null);
  mainWindow.loadURL('file://' + __dirname + '/static/index.html');

  mainWindow.on('closed', function() {
    mainWindow = null;
  });
  mainWindow.toggleDevTools();
});

app.on('error', function () {
  console.log(1234)
});



