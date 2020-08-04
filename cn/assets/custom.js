var nav_active = document.getElementsByClassName("md-nav__item--active");
if (nav_active[0].children[2].childElementCount == 0 || nav_active[0].children[2].children[1].childElementCount == 1) {
    document.getElementsByClassName("md-sidebar--primary")[0].style.display = "none"
    document.getElementsByClassName("md-content")[0].style.maxWidth = "100%"
}
