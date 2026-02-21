require(['gitbook', 'jQuery'], function (gitbook, $) {
    var PLUGIN = 'expandable-chapter-small2',
        TOGGLE_CLASSNAME = 'expanded',
        CHAPTER = '.chapter',
        // ARTICLES = '.articles',
        ARTICLES = '.chapter ul',
        FOLDABLE = '.chapter, .chapter li',
        ARTICLE_CHILDREN = 'ul',
        TRIGGER_TEMPLATE = '<i class="exc-trigger fa"></i>',
        LS_NAMESPACE = 'expChapters';

    var init = function () {
        // adding the trigger element to each ARTICLES parent and binding the event
        var config = gitbook.state.config.pluginsConfig || {};
        var articlesExpand = false;
        if (config && config[PLUGIN]) {
            articlesExpand = config[PLUGIN].articlesExpand || false;
        }
        if (articlesExpand) {
            $(ARTICLES)
                .parent(CHAPTER)
                .find(ARTICLE_CHILDREN)
                .prev()
                .css('cursor', 'pointer')
                .on('click', function (e) {
                    e.preventDefault();
                    e.stopPropagation();
                    toggle($(e.target).closest(FOLDABLE));
                })
                .append(TRIGGER_TEMPLATE);
        } else {
            $(ARTICLES)
                .parent(CHAPTER)
                .find(ARTICLE_CHILDREN)
                .prev()
                .append(
                    $(TRIGGER_TEMPLATE)
                        .on('click', function (e) {
                            e.preventDefault();
                            e.stopPropagation();
                            toggle($(e.target).closest(FOLDABLE));
                        })
                );
        }
        expand(lsItem());

        // expand current selected chapter with it's parents
        var activeChapter = $(CHAPTER + '.active');
        expand(activeChapter);

        // expand current selected chapter's children
        // expand(activeChapter.parents(CHAPTER));
        activeChapter.find(ARTICLE_CHILDREN).closest(FOLDABLE).each(function () {
            expand($(this));
        });
    }

    var toggle = function ($chapter) {
        if ($chapter.hasClass('expanded')) {
            collapse($chapter);
        } else {
            expand($chapter);
        }
    }

    var collapse = function ($chapter) {
        if ($chapter.length && $chapter.hasClass(TOGGLE_CLASSNAME)) {
            $chapter.removeClass(TOGGLE_CLASSNAME);
            lsItem($chapter);
        }
    }

    var expand = function ($chapter) {
        if ($chapter.length && !$chapter.hasClass(TOGGLE_CLASSNAME)) {
            $chapter.addClass(TOGGLE_CLASSNAME);
            lsItem($chapter);
        }
    }

    var lsItem = function () {
        var map = JSON.parse(localStorage.getItem(LS_NAMESPACE)) || {}
        if (arguments.length) {
            var $chapters = arguments[0];
            $chapters.each(function (index, element) {
                var level = $(this).data('level');
                var value = $(this).hasClass(TOGGLE_CLASSNAME);
                map[level] = value;
            })
            localStorage.setItem(LS_NAMESPACE, JSON.stringify(map));
        } else {
            return $(CHAPTER).map(function (index, element) {
                if (map[$(this).data('level')]) {
                    return this;
                }
            })
        }
    }

    gitbook.events.bind('page.change', function () {
        init()
    });
});
