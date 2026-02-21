// Enable footnote link support for pages with width < 1240.
//
function bind_footnote_links() {
    if ($(document).width() > 1240) {
        return;
    }
    let footnotes = $("div.footnotes").find("ol > li > p > a.reversefootnote");
    for (let i = 0; i < footnotes.length; i++) {
        let footnote = footnotes[i];
        footnote.addEventListener('click', function(e) {
            e.preventDefault();
            var target = $($(this).attr('href'));
            if (target.length) {
                $('div.body-inner').animate({
                    scrollTop: target.get(0).offsetTop,
                });
            }
        });
    }
}

if (document.readyState === "loading") {
    // Loading hasn't finished yet
    document.addEventListener("DOMContentLoaded", bind_footnote_links);
} else {
    // `DOMContentLoaded` has already fired
    bind_footnote_links();
}

