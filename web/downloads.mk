
WEB_DOWNLOADS := $(TMP_WEB_PATH)/_koheron.css
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.flot.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.flot.resize.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.flot.selection.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.flot.time.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.flot.axislabels.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.flot.canvas.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/bootstrap.min.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/bootstrap.min.css
WEB_DOWNLOADS += $(TMP_WEB_PATH)/jquery.min.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/centrale_lille.png
WEB_DOWNLOADS += $(TMP_WEB_PATH)/_koheron.png
WEB_DOWNLOADS += $(TMP_WEB_PATH)/kbird.ico
WEB_DOWNLOADS += $(TMP_WEB_PATH)/lato-v11-latin-400.woff2
WEB_DOWNLOADS += $(TMP_WEB_PATH)/lato-v11-latin-700.woff2
WEB_DOWNLOADS += $(TMP_WEB_PATH)/lato-v11-latin-900.woff2
WEB_DOWNLOADS += $(TMP_WEB_PATH)/glyphicons-halflings-regular.woff2
WEB_DOWNLOADS += $(TMP_WEB_PATH)/html-imports.min.js
WEB_DOWNLOADS += $(TMP_WEB_PATH)/html-imports.min.js.map
WEB_DOWNLOADS += $(TMP_WEB_PATH)/navigation.html

FLOT_VERSION = 0.8.3
#FLOT_VERSION = 4.2.6

$(TMP_WEB_PATH)/_koheron.css:
	mkdir -p $(@D)
	curl https://assets.koheron.com/css/main.css -o $@

$(TMP_WEB_PATH)/jquery.flot.js:
	mkdir -p $(@D)
	curl https://cdnjs.cloudflare.com/ajax/libs/flot/$(FLOT_VERSION)/jquery.flot.min.js -o $@

$(TMP_WEB_PATH)/jquery.flot.resize.js:
	mkdir -p $(@D)
	curl https://cdnjs.cloudflare.com/ajax/libs/flot/$(FLOT_VERSION)/jquery.flot.resize.min.js -o $@

$(TMP_WEB_PATH)/jquery.flot.selection.js:
	mkdir -p $(@D)
	curl https://cdnjs.cloudflare.com/ajax/libs/flot/$(FLOT_VERSION)/jquery.flot.selection.min.js -o $@

$(TMP_WEB_PATH)/jquery.flot.time.js:
	mkdir -p $(@D)
	curl https://cdnjs.cloudflare.com/ajax/libs/flot/$(FLOT_VERSION)/jquery.flot.time.min.js -o $@

$(TMP_WEB_PATH)/jquery.flot.axislabels.js:
	mkdir -p $(@D)
	curl https://raw.githubusercontent.com/markrcote/flot-axislabels/master/jquery.flot.axislabels.js -o $@

$(TMP_WEB_PATH)/jquery.flot.canvas.js:
	mkdir -p $(@D)
	curl https://cdnjs.cloudflare.com/ajax/libs/flot/$(FLOT_VERSION)/jquery.flot.canvas.js -o $@

$(TMP_WEB_PATH)/bootstrap.min.js:
	mkdir -p $(@D)
	curl http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js -o $@

$(TMP_WEB_PATH)/bootstrap.min.css:
	mkdir -p $(@D)
	curl http://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css -o $@

$(TMP_WEB_PATH)/jquery.min.js:
	mkdir -p $(@D)
	curl https://code.jquery.com/jquery-3.2.0.min.js -o $@

$(TMP_WEB_PATH)/_koheron.png:
	mkdir -p $(@D)
	curl https://www.centraliens-lille.org/medias/editor/oneshot-images/1645244191624d4e48b3ff4.png -o $@

$(TMP_WEB_PATH)/centrale_lille.png:
	mkdir -p $(@D)
	curl https://upload.wikimedia.org/wikipedia/commons/1/11/Logo_%C3%89cole_Centrale_de_Lille.png -o $@

$(TMP_WEB_PATH)/kbird.ico:
	mkdir -p $(@D)
	curl https://www.centraliens-lille.org/medias/editor/oneshot-images/1645244191624d4e48b3ff4.png -o $@

$(TMP_WEB_PATH)/lato-v11-latin-400.woff2:
	mkdir -p $(@D)
	curl https://fonts.gstatic.com/s/lato/v13/1YwB1sO8YE1Lyjf12WNiUA.woff2 -o $@

$(TMP_WEB_PATH)/lato-v11-latin-700.woff2:
	mkdir -p $(@D)
	curl https://fonts.gstatic.com/s/lato/v13/H2DMvhDLycM56KNuAtbJYA.woff2 -o $@

$(TMP_WEB_PATH)/lato-v11-latin-900.woff2:
	mkdir -p $(@D)
	curl https://fonts.gstatic.com/s/lato/v13/tI4j516nok_GrVf4dhunkg.woff2 -o $@

$(TMP_WEB_PATH)/glyphicons-halflings-regular.woff2:
	mkdir -p $(@D)
	curl https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/fonts/glyphicons-halflings-regular.woff2 -o $@

$(TMP_WEB_PATH)/html-imports.min.js:
	mkdir -p $(@D)
	curl https://raw.githubusercontent.com/webcomponents/html-imports/master/html-imports.min.js -o $@

$(TMP_WEB_PATH)/html-imports.min.js.map:
	mkdir -p $(@D)
	curl https://raw.githubusercontent.com/webcomponents/html-imports/master/html-imports.min.js.map -o $@

$(TMP_WEB_PATH)/navigation.html: $(WEB_PATH)/navigation.html
	mkdir -p $(@D)
	cp $< $@

$(WEB_PATH)/centrale_lille.png: $(TMP_WEB_PATH)/centrale_lille.png
	mkdir -p $(@D)
	cp $< $@
