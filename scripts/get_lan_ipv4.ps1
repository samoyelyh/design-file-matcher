$adapters = Get-NetIPConfiguration -ErrorAction SilentlyContinue |
    Where-Object { $_.NetAdapter.HardwareInterface -and $_.NetAdapter.Status -eq "Up" } |
    Sort-Object @{ Expression = { if ($_.IPv4DefaultGateway) { 0 } else { 1 } } },
                @{ Expression = { $_.NetIPv4Interface.InterfaceMetric } }

foreach ($adapter in $adapters) {
    $address = Get-NetIPAddress -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "169.254.*" } |
        Select-Object -ExpandProperty IPAddress -First 1
    if ($address) { return $address }
}
