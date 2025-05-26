use std::{any::TypeId, marker::PhantomData, ptr::NonNull};

use crate::{data_holder::DataHolder, DataReference};

/// A Genericized version of [DataReference]\<T\>
///
/// This type can be safely passed around without knowing the inner `T` of a [DataReference].  [DataReferenceGeneric] will track ref count just as [DataReference], however the inner data `T` cannot be accessed without first converting back into [DataReference]\<T\>.
///
/// Implements [From<DataReference<`T`>>]
pub struct DataReferenceGeneric {
	ptr: NonNull<()>,
	type_id: TypeId,
	inner_type_id: TypeId,
	drop_fn: &'static dyn Fn(NonNull<()>),
}

impl DataReferenceGeneric {
	/// The [TypeId] of the associated [DataReference]\<T\>, literally `TypeId::of::<DataReference<T>>()`
	pub fn type_id(&self) -> TypeId {
		self.type_id
	}
	/// The [TypeId] of the inner type T of the associated [DataReference]\<T\>.  If this [DataReferenceGeneric] is associated with [DataReference]\<T\>, this would be the result of `TypeId::of::<T>()`
	pub fn inner_type_id(&self) -> TypeId {
		self.inner_type_id
	}

	/// For a given type T, create a [DataReference]\<T\> if and only if T matches the type that was used to create this [DataReferenceGeneric], otherwise None
	///
	/// This increments the ref count if Some is returned
	pub fn to_typed<T: 'static>(&self) -> Option<DataReference<T>> {
		if TypeId::of::<DataReference<T>>() == self.type_id {
			let tmp = DataReference::<T> {
				ptr: self.ptr.cast::<DataHolder<T>>(),
				phantom: PhantomData,
			};

			tmp.increment_refcount();

			Some(tmp)
		} else {
			None
		}
	}
}

unsafe impl Send for DataReferenceGeneric {}
unsafe impl Sync for DataReferenceGeneric {}

impl<T: 'static> From<DataReference<T>> for DataReferenceGeneric {
	fn from(source: DataReference<T>) -> Self {
		source.increment_refcount();

		Self {
			ptr: source.ptr.cast::<()>(),
			type_id: TypeId::of::<DataReference<T>>(),
			inner_type_id: TypeId::of::<T>(),
			drop_fn: &DataReference::<T>::drop_impl,
		}
	}
}

impl Drop for DataReferenceGeneric {
	fn drop(&mut self) {
		(self.drop_fn)(self.ptr.cast::<()>());
	}
}
