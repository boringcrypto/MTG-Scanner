import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import { cardCount, fetchCount } from "../store.js";

export default {
  template: `
    <div class="row justify-content-center">
      <div class="col-md-5">

        <div class="card shadow-sm mb-3">
          <div class="card-body">
            <h5 class="card-title mb-4"><i class="bi bi-search me-2"></i>Card Lookup</h5>
            <form @submit.prevent="lookupByIdx">
              <div class="mb-3">
                <label class="form-label">Card index</label>
                <input v-model.number="idx" type="number" min="0" :max="cardCount ? cardCount - 1 : undefined"
                       class="form-control" placeholder="e.g. 42" autofocus required />
                <div class="form-text text-muted" v-if="cardCount !== null">
                  Database contains {{ cardCount.toLocaleString() }} cards
                  (0 – {{ (cardCount - 1).toLocaleString() }})
                </div>
              </div>
              <button class="btn btn-primary w-100" type="submit">
                <i class="bi bi-arrow-right-circle me-1"></i>View Card
              </button>
            </form>
          </div>
        </div>

        <div class="card shadow-sm mb-3">
          <div class="card-body">
            <h5 class="card-title mb-4"><i class="bi bi-upc-scan me-2"></i>Lookup by Product ID</h5>
            <form @submit.prevent="lookupByProductId">
              <div class="mb-3">
                <input v-model.trim="productId" type="text"
                       class="form-control" placeholder="e.g. 12345" required />
                <div class="form-text text-danger" v-if="productIdError">{{ productIdError }}</div>
              </div>
              <button class="btn btn-secondary w-100" type="submit">
                <i class="bi bi-arrow-right-circle me-1"></i>View Card
              </button>
            </form>
          </div>
        </div>

        <div class="d-flex flex-column gap-2">
          <router-link to="/duplicates" class="btn btn-outline-warning w-100">
            <i class="bi bi-copy me-2"></i>Browse Duplicate Groups
          </router-link>
          <router-link to="/sets" class="btn btn-outline-info w-100">
            <i class="bi bi-collection me-2"></i>Browse by Set
          </router-link>
          <router-link to="/compare" class="btn btn-outline-secondary w-100">
            <i class="bi bi-layout-split me-2"></i>Compare Two Cards
          </router-link>
        </div>

      </div>
    </div>
  `,
  setup() {
    const idx          = ref(0);
    const productId    = ref("");
    const productIdError = ref("");
    const router       = useRouter();
    onMounted(fetchCount);

    function lookupByIdx() {
      router.push(`/card/${idx.value}`);
    }

    async function lookupByProductId() {
      productIdError.value = "";
      try {
        const res = await fetch(`/api/card/by-product-id/${encodeURIComponent(productId.value)}`);
        if (!res.ok) { productIdError.value = `Not found: ${productId.value}`; return; }
        const data = await res.json();
        router.push(`/card/${data.index}`);
      } catch (e) {
        productIdError.value = "Request failed.";
      }
    }

    return { idx, cardCount, lookupByIdx, productId, productIdError, lookupByProductId };
  },
};
