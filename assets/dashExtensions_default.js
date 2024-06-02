window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng) {
            const windMarkerIcon = L.divIcon({
                className: `tc-marker`,
                html: `<img src='./assets/images/tc.png'/>`,
                iconSize: [50, 50]
            })
            const windMarker = L.marker(latlng, {
                icon: windMarkerIcon
            })
            const labelMarkerIcon = L.divIcon({
                className: `label-marker`,
                html: `<div>Max wind: ${feature.properties.ws}km/h</div>`
            })
            const labelMarker = L.marker(latlng, {
                icon: labelMarkerIcon
            })
            return L.layerGroup([windMarker, labelMarker])
        }
    }
});