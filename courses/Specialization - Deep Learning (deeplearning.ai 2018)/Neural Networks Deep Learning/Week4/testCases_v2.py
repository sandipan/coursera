
<!DOCTYPE HTML>
<html>

<head>
    <meta charset="utf-8">

    <title>testCases_v2.py (editing)</title>
    <link rel="shortcut icon" type="image/x-icon" href="/user/kuxvkrdrjorktiiqgjovch/static/base/images/favicon.ico?v=30780f272ab4aac64aa073a841546240">
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <link rel="stylesheet" href="/user/kuxvkrdrjorktiiqgjovch/static/components/jquery-ui/themes/smoothness/jquery-ui.min.css?v=9b2c8d3489227115310662a343fce11c" type="text/css" />
    <link rel="stylesheet" href="/user/kuxvkrdrjorktiiqgjovch/static/components/jquery-typeahead/dist/jquery.typeahead.min.css?v=7afb461de36accb1aa133a1710f5bc56" type="text/css" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    
<link rel="stylesheet" href="/user/kuxvkrdrjorktiiqgjovch/static/components/codemirror/lib/codemirror.css?v=2336fb49f85e9fa887ada9af35223dce">
<link rel="stylesheet" href="/user/kuxvkrdrjorktiiqgjovch/static/components/codemirror/addon/dialog/dialog.css?v=c89dce10b44d2882a024e7befc2b63f5">

    <link rel="stylesheet" href="/user/kuxvkrdrjorktiiqgjovch/static/style/style.min.css?v=f6c09475baf6beabd41f8fe518601204" type="text/css"/>
    

    <link rel="stylesheet" href="/user/kuxvkrdrjorktiiqgjovch/custom/custom.css" type="text/css" />
    <script src="/user/kuxvkrdrjorktiiqgjovch/static/components/es6-promise/promise.min.js?v=f004a16cb856e0ff11781d01ec5ca8fe" type="text/javascript" charset="utf-8"></script>
    <script src="/user/kuxvkrdrjorktiiqgjovch/static/components/requirejs/require.js?v=6da8be361b9ee26c5e721e76c6d4afce" type="text/javascript" charset="utf-8"></script>
    <script>
      require.config({
          
          urlArgs: "v=20170830063204",
          
          baseUrl: '/user/kuxvkrdrjorktiiqgjovch/static/',
          paths: {
            'auth/js/main': 'auth/js/main.min',
            custom : '/user/kuxvkrdrjorktiiqgjovch/custom',
            nbextensions : '/user/kuxvkrdrjorktiiqgjovch/nbextensions',
            kernelspecs : '/user/kuxvkrdrjorktiiqgjovch/kernelspecs',
            underscore : 'components/underscore/underscore-min',
            backbone : 'components/backbone/backbone-min',
            jquery: 'components/jquery/jquery.min',
            bootstrap: 'components/bootstrap/js/bootstrap.min',
            bootstraptour: 'components/bootstrap-tour/build/js/bootstrap-tour.min',
            'jquery-ui': 'components/jquery-ui/ui/minified/jquery-ui.min',
            moment: 'components/moment/moment',
            codemirror: 'components/codemirror',
            termjs: 'components/term.js/src/term',
            typeahead: 'components/jquery-typeahead/dist/jquery.typeahead'
          },
	  map: { // for backward compatibility
	    "*": {
		"jqueryui": "jquery-ui",
	    }
	  },
          shim: {
            typeahead: {
              deps: ["jquery"],
              exports: "typeahead"
            },
            underscore: {
              exports: '_'
            },
            backbone: {
              deps: ["underscore", "jquery"],
              exports: "Backbone"
            },
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
            bootstraptour: {
              deps: ["bootstrap"],
              exports: "Tour"
            },
            "jquery-ui": {
              deps: ["jquery"],
              exports: "$"
            }
          },
          waitSeconds: 30,
      });

      require.config({
          map: {
              '*':{
                'contents': 'services/contents',
              }
          }
      });

      define("bootstrap", function () {
          return window.$;
      });

      define("jquery", function () {
          return window.$;
      });

      define("jqueryui", function () {
          return window.$;
      });

      define("jquery-ui", function () {
          return window.$;
      });
      // error-catching custom.js shim.
      define("custom", function (require, exports, module) {
          try {
              var custom = require('custom/custom');
              console.debug('loaded custom.js');
              return custom;
          } catch (e) {
              console.error("error loading custom.js", e);
              return {};
          }
      })
    </script>

    
    

</head>

<body class="edit_app " 
data-base-url="/user/kuxvkrdrjorktiiqgjovch/"
data-file-path="Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/testCases_v2.py"

>

<noscript>
    <div id='noscript'>
      Jupyter Notebook requires JavaScript.<br>
      Please enable it to proceed.
  </div>
</noscript>

<div id="header">
  <div id="header-container" class="container">
  <div id="ipython_notebook" class="nav navbar-brand pull-left"><a href="/user/kuxvkrdrjorktiiqgjovch/tree" title='dashboard'>
<img src='/hub/logo' alt='Jupyter Notebook'/>
</a></div>

  

  
  

    <span id="login_widget">
      
        <button id="logout" class="btn btn-sm navbar-btn">Logout</button>
      
    </span>

  

  

<a href='/hub/home'
 class='btn btn-default btn-sm navbar-btn pull-right'
 style='margin-right: 4px; margin-left: 2px;'
>
Control Panel</a>


  

<span id="save_widget" class="pull-left save_widget">
    <span class="filename"></span>
    <span class="last_modified"></span>
</span>


  </div>
  <div class="header-bar"></div>

  

<div id="menubar-container" class="container">
  <div id="menubar">
    <div id="menus" class="navbar navbar-default" role="navigation">
      <div class="container-fluid">
          <p  class="navbar-text indicator_area">
          <span id="current-mode" >current mode</span>
          </p>
        <button type="button" class="btn btn-default navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
          <i class="fa fa-bars"></i>
          <span class="navbar-text">Menu</span>
        </button>
        <ul class="nav navbar-nav navbar-right">
          <li id="notification_area"></li>
        </ul>
        <div class="navbar-collapse collapse">
          <ul class="nav navbar-nav">
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">File</a>
              <ul id="file-menu" class="dropdown-menu">
                <li id="new-file"><a href="#">New</a></li>
                <li id="save-file"><a href="#">Save</a></li>
                <li id="rename-file"><a href="#">Rename</a></li>
                <li id="download-file"><a href="#">Download</a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Edit</a>
              <ul id="edit-menu" class="dropdown-menu">
                <li id="menu-find"><a href="#">Find</a></li>
                <li id="menu-replace"><a href="#">Find &amp; Replace</a></li>
                <li class="divider"></li>
                <li class="dropdown-header">Key Map</li>
                <li id="menu-keymap-default"><a href="#">Default<i class="fa"></i></a></li>
                <li id="menu-keymap-sublime"><a href="#">Sublime Text<i class="fa"></i></a></li>
                <li id="menu-keymap-vim"><a href="#">Vim<i class="fa"></i></a></li>
                <li id="menu-keymap-emacs"><a href="#">emacs<i class="fa"></i></a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">View</a>
              <ul id="view-menu" class="dropdown-menu">
              <li id="toggle_header" title="Show/Hide the logo and notebook title (above menu bar)">
              <a href="#">Toggle Header</a></li>
              <li id="menu-line-numbers"><a href="#">Toggle Line Numbers</a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Language</a>
              <ul id="mode-menu" class="dropdown-menu">
              </ul>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="lower-header-bar"></div>


</div>

<div id="site">


<div id="texteditor-backdrop">
<div id="texteditor-container" class="container"></div>
</div>


</div>






    



    <script src="/user/kuxvkrdrjorktiiqgjovch/static/edit/js/main.min.js?v=cc7536522bdf28a7f0059136b696fd6e" type="text/javascript" charset="utf-8"></script>



</body>

</html>