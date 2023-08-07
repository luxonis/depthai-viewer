// NOTE: This file was autogenerated by re_types_builder; DO NOT EDIT.
// Based on "crates/re_types/definitions/rerun/datatypes/quaternion.fbs"

#pragma once

#include <arrow/type_fwd.h>
#include <cstdint>

namespace rerun {
    namespace datatypes {
        /// A Quaternion represented by 4 real numbers.
        struct Quaternion {
            float xyzw[4];

          public:
            Quaternion() = default;

            /// Returns the arrow data type this type corresponds to.
            static const std::shared_ptr<arrow::DataType>& to_arrow_datatype();

            /// Creates a new array builder with an array of this type.
            static arrow::Result<std::shared_ptr<arrow::FixedSizeListBuilder>>
                new_arrow_array_builder(arrow::MemoryPool* memory_pool);

            /// Fills an arrow array builder with an array of this type.
            static arrow::Status fill_arrow_array_builder(
                arrow::FixedSizeListBuilder* builder, const Quaternion* elements,
                size_t num_elements
            );
        };
    } // namespace datatypes
} // namespace rerun