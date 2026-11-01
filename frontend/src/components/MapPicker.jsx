import React, { useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';

import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl,
});

// Component to handle map clicks and marker updates
function MapClickHandler({ onLocationChange }) {
  useMapEvents({
    click(e) {
      onLocationChange(
        parseFloat(e.latlng.lat.toFixed(5)),
        parseFloat(e.latlng.lng.toFixed(5))
      );
    },
  });
  return null;
}

// Component to fly map to new center when district changes
function MapRecenter({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center && center[0] && center[1]) {
      map.flyTo(center, 9, { duration: 1.2 });
    }
  }, [center, map]);
  return null;
}

export default function MapPicker({
  latitude,
  longitude,
  districtCentroid,
  districtName,
  onLocationChange,
}) {
  const position = [latitude, longitude];
  const centroidPos = districtCentroid ? [districtCentroid.lat, districtCentroid.lon] : position;

  const eventHandlers = useMemo(
    () => ({
      dragend(e) {
        const marker = e.target;
        if (marker != null) {
          const latLng = marker.getLatLng();
          onLocationChange(
            parseFloat(latLng.lat.toFixed(5)),
            parseFloat(latLng.lng.toFixed(5))
          );
        }
      },
    }),
    [onLocationChange]
  );

  return (
    <div className="relative w-full h-[380px] rounded-xl overflow-hidden border border-slate-200 shadow-sm bg-slate-100">
      <MapContainer
        center={position}
        zoom={9}
        scrollWheelZoom={true}
        className="w-full h-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <MapClickHandler onLocationChange={onLocationChange} />
        <MapRecenter center={centroidPos} />

        {/* Selected Coordinate Marker (Draggable) */}
        <Marker position={position} draggable={true} eventHandlers={eventHandlers}>
          <Popup>
            <div className="text-xs">
              <p className="font-bold text-emerald-800">Selected Location</p>
              <p>Lat: {latitude.toFixed(4)}</p>
              <p>Lon: {longitude.toFixed(4)}</p>
              <p className="text-slate-500 italic mt-1">Drag marker to adjust</p>
            </div>
          </Popup>
        </Marker>

        {/* Valid Catchment Circle around District Centroid (65km) */}
        {districtCentroid && (
          <Circle
            center={[districtCentroid.lat, districtCentroid.lon]}
            radius={65000}
            pathOptions={{
              color: '#059669',
              fillColor: '#10b981',
              fillOpacity: 0.08,
              dashArray: '4, 6',
            }}
          >
            <Popup>
              <div className="text-xs">
                <span className="font-semibold text-emerald-700">{districtName} Catchment Zone</span>
                <p className="text-slate-500">Radius: ~65 km around official centroid</p>
              </div>
            </Popup>
          </Circle>
        )}
      </MapContainer>

      {/* Floating map hint */}
      <div className="absolute bottom-3 left-3 z-[400] bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 shadow-sm border border-slate-200/80 pointer-events-none">
        📍 Click map or drag pin to adjust coordinates
      </div>
    </div>
  );
}
