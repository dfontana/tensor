local native = require("tensor_native")

local function values(length, multiplier, offset, modulus)
    local result = {}
    for index = 1, length do
        result[index] = (index * multiplier + offset) % modulus
    end
    return result
end

local function verify(native_output, table_output, lhs, rhs, length)
    for index = 1, length do
        local expected = lhs[index] + rhs[index]
        assert(native_output[index] == expected,
            string.format("native output mismatch at index %d", index))
        assert(table_output[index] == expected,
            string.format("table output mismatch at index %d", index))
    end
end

for _, length in ipairs({1024, 1000003}) do
    local lhs = values(length, 3, 7, 97)
    local rhs = values(length, 5, 11, 89)
    local native_lhs = native.storage(lhs)
    local native_rhs = native.storage(rhs)
    local native_output = native.storage_zeros(length)
    local table_output = {}

    local native_start = os.clock()
    native.add_into(native_lhs, native_rhs, native_output)
    local native_seconds = os.clock() - native_start

    local table_start = os.clock()
    for index = 1, length do
        table_output[index] = lhs[index] + rhs[index]
    end
    local table_seconds = os.clock() - table_start

    verify(native_output, table_output, lhs, rhs, length)
    print(string.format(
        "length=%d native_storage_add=%.6fs lua_table_add=%.6fs verified",
        length, native_seconds, table_seconds))
end
