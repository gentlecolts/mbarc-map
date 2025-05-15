use std::sync::{
	atomic::{AtomicBool, AtomicUsize, Ordering},
	Arc, Mutex,
};

use crate::fixed_address_continuous_allocation::{FaVec, FaVecIndex};

//Note: at time of writing, it does not make sense to expose this value to the end user
//This greatly complicates the signature, interferes with type inference, and makes DataReferenceGeneric more ugly (requiring it to be a template too)
//it ultimately wouldn't be that hard to expose this, but there's little value to be gained until proven otherwise
//in a perfect world, this value could be set via some user-defined cfg at compile time, as it should likely only matter on a platform basis anyways
//in which case, a potential future TODO would be configuring this on a per platform basis, or do more work in finding a smarter mechanism for choosing this value, if it's found that this is significant enough to matter
//TODO: additionally, determine whether or not it's even worth exposing this, as it feels like doing so could encourage bad designs more so than support good ones
//TODO: consider setting this via option_env! macro, giving an out-of-the-way means of configuring this if someone really needed to
pub(crate) const DATA_HOLDER_BLOCK_SIZE_INTERNAL: usize = 32;

pub(crate) type SharedDataContainerType<T> =
	Arc<Mutex<FaVec<DataHolder<T>, DATA_HOLDER_BLOCK_SIZE_INTERNAL>>>;

pub(crate) struct DataHolder<T> {
	pub(crate) ref_count: AtomicUsize,
	pub(crate) pending_removal: AtomicBool,

	//TODO: having these here rather than in DataReference means less duplication, but also makes for fatter entries in the FaVec array
	//the deduplication is probably worth it, but will need to bench for it being worth the potentially less good caching behavior
	pub(crate) owner: SharedDataContainerType<T>,
	pub(crate) owning_key: FaVecIndex<32>,

	pub data: Mutex<T>,
}

impl<T> DataHolder<T> {
	pub(crate) fn deleted(&self) -> bool {
		//TODO: evaluate safety of this ordering
		self.pending_removal.load(Ordering::Acquire)
	}

	pub(crate) fn ref_count(&self) -> usize {
		//TODO: evaluate safety of this ordering
		self.ref_count.load(Ordering::Acquire)
	}

	pub(crate) fn set_deleted(&self) {
		//TODO: evaluate safety of this ordering
		self.pending_removal.store(true, Ordering::Release);
	}

	pub(crate) fn increment_refcount(&self) -> usize {
		let old_rc = self.ref_count.fetch_add(1, Ordering::Relaxed);
		if old_rc >= isize::MAX as usize {
			std::process::abort();
		}
		old_rc
	}
}
