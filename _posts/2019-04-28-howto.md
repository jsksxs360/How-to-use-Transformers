---
title: How to Get Started
author: Tao He
date: 2019-04-28
category: Jekyll
layout: post
---

The jekyll-gitbook theme can be used just as other [Jekyll themes][3] and
support [remote theme][2] on [Github Pages][1], see [the official guide][4]
as well.

You can introduce this jekyll theme into your own site by either

- [Fork][5] this repository and add your markdown posts to the `_posts` folder, then
  push to your own Github repository.
- Use as a remote theme in your [`_config.yml`][6](just like what we do for this
  site itself),

```yaml
# Configurations
title:            Jekyll Gitbook
longtitle:        Jekyll Gitbook

remote_theme:     sighingnow/jekyll-gitbook
```

> ##### TIP
>
> No need to push generated HTML bundle.
{: .block-tip }

[1]: https://pages.github.com
[2]: https://github.com/sighingnow/jekyll-gitbook/fork
[3]: https://pages.github.com/themes
[4]: https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/adding-a-theme-to-your-github-pages-site-using-jekyll
[5]: https://github.com/sighingnow/jekyll-gitbook/fork
[6]: https://github.com/sighingnow/jekyll-gitbook/blob/master/_config.yml
