    "config": {
        "plugins": ["fontsettings", "highlight", "livereload", "lunr", "search", "sharing", "theme-default", "livereload"],
        "styles": {
            "ebook": "styles/ebook.css",
            "epub": "styles/epub.css",
            "mobi": "styles/mobi.css",
            "pdf": "styles/pdf.css",
            "print": "styles/print.css",
            "website": "styles/website.css"
        },
        "pluginsConfig": {
            "expandable-chapter-small2": {
                "articlesExpand": true,
            },
            "fontsettings": {
                "family": "sans",
                "size": 2,
                "theme": "white"
            },
            "highlight": {},
            "livereload": {},
            "lunr": {
                "ignoreSpecialCharacters": false,
                "maxIndexSize": 1000000
            },
            "search": {},

            {%- include gitbook-sharing.json.tpl -%}

            "theme-default": {
                "showLevel": false,
                "styles": {
                    "ebook": "styles/ebook.css",
                    "epub": "styles/epub.css",
                    "mobi": "styles/mobi.css",
                    "pdf": "styles/pdf.css",
                    "print": "styles/print.css",
                    "website": "styles/website.css"
                }
            },
        },
        "theme": "default",
        "author": "Tao He",
        "pdf": {
            "pageNumbers": true,
            "fontSize": 12,
            "fontFamily": "Arial",
            "paperSize": "a4",
            "chapterMark": "pagebreak",
            "pageBreaksBefore": "/",
            "margin": {
                "right": 62,
                "left": 62,
                "top": 56,
                "bottom": 56
            }
        },
        "structure": {
            "langs": "LANGS.md",
            "readme": "README.md",
        },
        "variables": {},
        "title": "{{site.title}}",
        "language": "en",
        "gitbook": "*"
    },
    "file": {
        "path": "{{ page.path }}",
        "mtime": "{{ page.date }}",
        "type": "markdown"
    },
    "gitbook": {
        "version": "{{site.gitbook_version}}",
        "time": "{{site.time}}"
    },
    "basePath": "{{site.baseurl}}",
    "book": {
        "language": ""
    }
