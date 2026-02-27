import { createRouter, createWebHashHistory } from "vue-router";
import HomeView       from "./views/HomeView.js";
import CardView       from "./views/CardView.js";
import DuplicatesView from "./views/DuplicatesView.js";
import SetsView       from "./views/SetsView.js";
import SetView        from "./views/SetView.js";

export default createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/",             component: HomeView       },
    { path: "/card/:idx",    component: CardView       },
    { path: "/duplicates",   component: DuplicatesView },
    { path: "/sets",         component: SetsView       },
    { path: "/set/:set_id",  component: SetView        },
  ],
});
